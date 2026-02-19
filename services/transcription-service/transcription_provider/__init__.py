import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import numpy as np

try:
    import websockets
except Exception:  # pragma: no cover - validated in runtime path
    websockets = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - type hint compatibility for tests without deps
    WhisperModel = Any

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {
    "local",
    "openai",
    "assemblyai",
    "deepgram",
    "revai",
    "gcp",
    "telnyx",
    "soniox",
    "elevenlabs",
}

PROVIDER_ENV_KEY_FALLBACKS: Dict[str, List[str]] = {
    "openai": ["REMOTE_TRANSCRIBER_API_KEY", "OPENAI_API_KEY"],
    "assemblyai": ["ASSEMBLYAI_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "deepgram": ["DEEPGRAM_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "revai": ["REVAI_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "gcp": ["GCP_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "telnyx": ["TELNYX_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "soniox": ["SONIOX_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
    "elevenlabs": ["ELEVENLABS_API_KEY", "REMOTE_TRANSCRIBER_API_KEY"],
}

# These defaults are intentionally conservative and configurable.
# For non-OpenAI providers, production deployments should override ws_url/messages/events
# in TRANSCRIBER_PROVIDER_CONFIG_JSON as provider APIs evolve.
PROVIDER_WEBSOCKET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "assemblyai": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "",
        "query": {"sample_rate": "${sample_rate_hz}"},
        "messages": {
            "append": {"audio_data": "${audio_b64}"},
            "commit": {"terminate_session": True},
        },
        "events": {
            "delta_types": ["partial_transcript"],
            "final_types": ["final_transcript"],
            "delta_field": "text",
            "final_field": "text",
            "language_field": "language",
            "error_types": ["error"],
            "error_field": "error",
        },
    },
    "deepgram": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Token ",
        "query": {"encoding": "linear16", "sample_rate": "${sample_rate_hz}", "channels": 1},
        "messages": {
            "append": {"audio": "${audio_b64}"},
            "commit": {"type": "Finalize"},
        },
        "events": {
            "delta_types": ["Results"],
            "final_types": ["Results"],
            "delta_field": "channel.alternatives.0.transcript",
            "final_field": "channel.alternatives.0.transcript",
            "language_field": "channel.alternatives.0.language",
            "final_flag_field": "is_final",
            "error_types": ["Error"],
            "error_field": "description",
        },
    },
    "revai": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Bearer ",
        "query": {"content_type": "audio/x-raw;layout=interleaved;rate=${sample_rate_hz};format=S16LE;channels=1"},
        "messages": {
            "append": {"audio_data": "${audio_b64}"},
            "commit": {"event": "stop"},
        },
        "events": {
            "delta_types": ["partial"],
            "final_types": ["final"],
            "delta_field": "elements_text",
            "final_field": "elements_text",
            "language_field": "language",
            "error_types": ["error"],
            "error_field": "detail",
        },
    },
    "gcp": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Bearer ",
        "query": {},
        "messages": {
            "session": {
                "streaming_config": {
                    "config": {
                        "encoding": "LINEAR16",
                        "sample_rate_hertz": "${sample_rate_hz}",
                        "language_code": "${language_or_en}",
                    },
                    "interim_results": True,
                }
            },
            "append": {"audio_content": "${audio_b64}"},
            "commit": {"finish": True},
        },
        "events": {
            "delta_types": ["results"],
            "final_types": ["results"],
            "delta_field": "results.0.alternatives.0.transcript",
            "final_field": "results.0.alternatives.0.transcript",
            "language_field": "results.0.language_code",
            "final_flag_field": "results.0.is_final",
            "error_types": ["error"],
            "error_field": "error.message",
        },
    },
    "telnyx": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Bearer ",
        "query": {},
        "messages": {
            "append": {"audio": "${audio_b64}"},
            "commit": {"type": "stop"},
        },
        "events": {
            "delta_types": ["transcription.partial"],
            "final_types": ["transcription.final"],
            "delta_field": "text",
            "final_field": "text",
            "language_field": "language",
            "error_types": ["error"],
            "error_field": "message",
        },
    },
    "soniox": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Bearer ",
        "query": {},
        "messages": {
            "append": {"audio": "${audio_b64}"},
            "commit": {"type": "finalize"},
        },
        "events": {
            "delta_types": ["partial_transcript"],
            "final_types": ["final_transcript"],
            "delta_field": "text",
            "final_field": "text",
            "language_field": "language",
            "error_types": ["error"],
            "error_field": "message",
        },
    },
    "elevenlabs": {
        "ws_url": "",
        "sample_rate_hz": 16000,
        "auth_header_name": "xi-api-key",
        "auth_header_prefix": "",
        "query": {},
        "messages": {
            "append": {"audio": "${audio_b64}"},
            "commit": {"type": "finalize"},
        },
        "events": {
            "delta_types": ["transcript.partial"],
            "final_types": ["transcript.final"],
            "delta_field": "text",
            "final_field": "text",
            "language_field": "language",
            "error_types": ["error"],
            "error_field": "message",
        },
    },
}


@dataclass
class ProviderRequest:
    language: Optional[str]
    prompt: Optional[str]
    task: str
    transcription_tier: str
    requested_model: str
    temperature: float
    sample_rate: int


@dataclass
class ProviderResult:
    text: str
    language: str
    duration: float
    segments: List[Dict[str, Any]]


class TranscriptionProvider:
    async def transcribe(self, audio_array: np.ndarray, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError


class LocalWhisperProvider(TranscriptionProvider):
    def __init__(
        self,
        *,
        model: WhisperModel,
        executor,
        beam_size: int,
        best_of: int,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        condition_on_previous_text: bool,
        prompt_reset_on_temperature: float,
        vad_filter: bool,
        vad_filter_threshold: float,
        vad_min_silence_duration_ms: int,
        use_temperature_fallback: bool,
        temperature_fallback_chain: List[float],
    ):
        self.model = model
        self.executor = executor
        self.beam_size = beam_size
        self.best_of = best_of
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.prompt_reset_on_temperature = prompt_reset_on_temperature
        self.vad_filter = vad_filter
        self.vad_filter_threshold = vad_filter_threshold
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.use_temperature_fallback = use_temperature_fallback
        self.temperature_fallback_chain = temperature_fallback_chain

    def _looks_like_silence(self, segments: List[Dict[str, Any]]) -> bool:
        if not segments:
            return True
        for seg in segments:
            if not (
                float(seg.get("no_speech_prob", 0.0)) > self.no_speech_threshold
                and float(seg.get("avg_logprob", 0.0)) < self.log_prob_threshold
            ):
                return False
        return True

    def _looks_like_hallucination(self, segments: List[Dict[str, Any]]) -> bool:
        for seg in segments:
            if float(seg.get("compression_ratio", 0.0)) > self.compression_ratio_threshold:
                return True
            if float(seg.get("avg_logprob", 0.0)) < self.log_prob_threshold:
                return True
        return False

    async def transcribe(self, audio_array: np.ndarray, request: ProviderRequest) -> ProviderResult:
        temps = self.temperature_fallback_chain if self.use_temperature_fallback else [request.temperature]

        best: Optional[Tuple[str, str, float, List[Dict[str, Any]]]] = None
        last_info = None
        last_segments: List[Dict[str, Any]] = []

        for temp in temps:

            def _transcribe_sync():
                return self.model.transcribe(
                    audio_array,
                    language=request.language,
                    task=request.task,
                    initial_prompt=request.prompt,
                    temperature=temp,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    log_prob_threshold=self.log_prob_threshold,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=self.condition_on_previous_text,
                    prompt_reset_on_temperature=self.prompt_reset_on_temperature,
                    vad_filter=self.vad_filter,
                    vad_parameters={
                        "threshold": self.vad_filter_threshold,
                        "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                    },
                    word_timestamps=False,
                )

            segments_iter, info = await asyncio.get_event_loop().run_in_executor(self.executor, _transcribe_sync)
            last_info = info

            segments: List[Dict[str, Any]] = []
            for idx, segment in enumerate(segments_iter):
                segments.append(
                    {
                        "id": idx,
                        "seek": 0,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "tokens": [],
                        "temperature": temp,
                        "avg_logprob": segment.avg_logprob,
                        "compression_ratio": segment.compression_ratio,
                        "no_speech_prob": segment.no_speech_prob,
                        "audio_start": segment.start,
                        "audio_end": segment.end,
                    }
                )
            last_segments = segments

            if self._looks_like_silence(segments):
                best = ("", info.language, 0.0, [])
                break

            if not self._looks_like_hallucination(segments):
                full_text = " ".join([s["text"].strip() for s in segments]).strip()
                duration = segments[-1]["end"] if segments else 0.0
                best = (full_text, info.language, duration, segments)
                break

        if best is None:
            info = last_info
            segments = last_segments
            full_text = " ".join([s["text"].strip() for s in segments]).strip()
            duration = segments[-1]["end"] if segments else 0.0
            detected_language = info.language if info else (request.language or "unknown")
            best = (full_text, detected_language, duration, segments)

        full_text, detected_language, duration, segments = best
        return ProviderResult(
            text=full_text,
            language=detected_language,
            duration=duration,
            segments=segments,
        )


class RealtimeWebsocketProvider(TranscriptionProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        api_key: str,
        ws_url: str,
        model: str,
        timeout_s: int,
        sample_rate_hz: int,
        auth_header_name: str,
        auth_header_prefix: str,
        query: Dict[str, Any],
        messages: Dict[str, Any],
        events: Dict[str, Any],
    ):
        self.provider_name = provider_name
        self.api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self.timeout_s = timeout_s
        self.sample_rate_hz = sample_rate_hz
        self.auth_header_name = auth_header_name
        self.auth_header_prefix = auth_header_prefix
        self.query = query
        self.messages = messages
        self.events = events

    @staticmethod
    def _resample_mono(audio_array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate <= 0 or source_rate == target_rate:
            return audio_array
        src_len = len(audio_array)
        if src_len == 0:
            return audio_array
        dst_len = max(1, int(round(src_len * target_rate / source_rate)))
        src_idx = np.linspace(0, src_len - 1, num=src_len, dtype=np.float64)
        dst_idx = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float64)
        return np.interp(dst_idx, src_idx, audio_array).astype(np.float32)

    @staticmethod
    def _as_pcm16_bytes(audio_array: np.ndarray) -> bytes:
        clipped = np.clip(audio_array, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def _build_ws_url(self, context: Dict[str, Any]) -> str:
        rendered_query = _render_template(self.query, context) if self.query else {}
        parsed = urlparse(self.ws_url)
        existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for key, value in rendered_query.items():
            existing[str(key)] = str(value)
        query_str = urlencode(existing, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query_str, parsed.fragment))

    def _build_headers(self) -> Dict[str, str]:
        value = f"{self.auth_header_prefix}{self.api_key}" if self.auth_header_prefix is not None else self.api_key
        return {self.auth_header_name: value}

    async def _transcribe_via_ws(self, audio_array: np.ndarray, request: ProviderRequest) -> Tuple[str, Optional[str]]:
        if websockets is None:
            raise RuntimeError("websockets package is required for realtime providers")

        resampled = self._resample_mono(audio_array, request.sample_rate, self.sample_rate_hz)
        audio_payload = base64.b64encode(self._as_pcm16_bytes(resampled)).decode("ascii")

        context = {
            "audio_b64": audio_payload,
            "model": self.model,
            "language": request.language or "",
            "language_or_en": request.language or "en",
            "prompt": request.prompt or "",
            "sample_rate_hz": self.sample_rate_hz,
            "requested_model": request.requested_model,
            "task": request.task,
            "tier": request.transcription_tier,
        }

        ws_url = self._build_ws_url(context)
        headers = self._build_headers()

        transcript_final: Optional[str] = None
        transcript_deltas: List[str] = []
        detected_language: Optional[str] = None

        async with websockets.connect(ws_url, additional_headers=headers, open_timeout=self.timeout_s) as ws:
            session_message = self.messages.get("session")
            if session_message:
                await ws.send(json.dumps(_render_template(session_message, context)))

            append_message = self.messages.get("append")
            if append_message is None:
                raise RuntimeError(f"Provider '{self.provider_name}' requires messages.append config")
            await ws.send(json.dumps(_render_template(append_message, context)))

            commit_message = self.messages.get("commit")
            if commit_message:
                await ws.send(json.dumps(_render_template(commit_message, context)))

            while True:
                try:
                    raw_event = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(f"Timed out waiting for {self.provider_name} transcription response") from exc

                event = json.loads(raw_event)
                event_type = _event_type(event)

                if event_type in set(self.events.get("error_types", [])):
                    err = _nested_get(event, self.events.get("error_field", "error.message"))
                    message = err if isinstance(err, str) and err.strip() else str(event)
                    raise RuntimeError(f"{self.provider_name} realtime API error: {message}")

                delta_types = set(self.events.get("delta_types", []))
                final_types = set(self.events.get("final_types", []))

                if event_type in delta_types:
                    delta_val = _nested_get(event, self.events.get("delta_field", "delta"))
                    if isinstance(delta_val, str) and delta_val:
                        transcript_deltas.append(delta_val)

                if event_type in final_types:
                    final_flag_path = self.events.get("final_flag_field")
                    is_final = True
                    if final_flag_path:
                        is_final = bool(_nested_get(event, final_flag_path))
                    if is_final:
                        final_val = _nested_get(event, self.events.get("final_field", "text"))
                        transcript_final = final_val if isinstance(final_val, str) else ""
                        lang_val = _nested_get(event, self.events.get("language_field", "language"))
                        if isinstance(lang_val, str) and lang_val:
                            detected_language = lang_val
                        break

                if event_type in set(self.events.get("end_types", [])):
                    break

        transcript_text = transcript_final if transcript_final is not None else "".join(transcript_deltas)
        return transcript_text.strip(), detected_language

    async def transcribe(self, audio_array: np.ndarray, request: ProviderRequest) -> ProviderResult:
        text, detected_language = await self._transcribe_via_ws(audio_array, request)
        duration = float(len(audio_array)) / float(max(request.sample_rate, 1))
        language = request.language or detected_language or "unknown"

        segments: List[Dict[str, Any]] = []
        if text:
            segments = [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "tokens": [],
                    "temperature": request.temperature,
                    "avg_logprob": -0.5,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.1,
                    "audio_start": 0.0,
                    "audio_end": duration,
                }
            ]

        return ProviderResult(
            text=text,
            language=language,
            duration=duration,
            segments=segments,
        )


class OpenAIRealtimeProvider(RealtimeWebsocketProvider):
    def __init__(self, *, api_key: str, model: str, base_url: str, timeout_s: int):
        ws_url = _normalize_base_url_to_ws(base_url, "/realtime")
        super().__init__(
            provider_name="openai",
            api_key=api_key,
            ws_url=ws_url,
            model=model,
            timeout_s=timeout_s,
            sample_rate_hz=24000,
            auth_header_name="Authorization",
            auth_header_prefix="Bearer ",
            query={"intent": "transcription", "model": "${model}"},
            messages={
                "session": {
                    "type": "transcription_session.update",
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "${model}",
                        "language": "${language}",
                        "prompt": "${prompt}",
                    },
                    "turn_detection": None,
                },
                "append": {"type": "input_audio_buffer.append", "audio": "${audio_b64}"},
                "commit": {"type": "input_audio_buffer.commit"},
            },
            events={
                "delta_types": [
                    "conversation.item.input_audio_transcription.delta",
                    "response.audio_transcript.delta",
                ],
                "final_types": [
                    "conversation.item.input_audio_transcription.completed",
                    "response.audio_transcript.done",
                ],
                "delta_field": "delta",
                "final_field": "transcript",
                "language_field": "language",
                "error_types": ["error"],
                "error_field": "error.message",
            },
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers["OpenAI-Beta"] = "realtime=v1"
        return headers

    async def _transcribe_via_ws(self, audio_array: np.ndarray, request: ProviderRequest) -> Tuple[str, Optional[str]]:
        # Remove empty language/prompt fields to avoid upstream validation noise.
        if self.messages.get("session"):
            session = dict(self.messages["session"])
            input_cfg = dict(session.get("input_audio_transcription", {}))
            if not request.language:
                input_cfg.pop("language", None)
            if not request.prompt:
                input_cfg.pop("prompt", None)
            session["input_audio_transcription"] = input_cfg
            original = self.messages.get("session")
            self.messages["session"] = session
            try:
                return await super()._transcribe_via_ws(audio_array, request)
            finally:
                self.messages["session"] = original
        return await super()._transcribe_via_ws(audio_array, request)


def _nested_get(payload: Any, path: str) -> Any:
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
            except Exception:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
        if current is None:
            return None
    return current


def _event_type(event: Dict[str, Any]) -> str:
    for key in ("type", "event", "message_type"):
        val = event.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _render_template(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, val in context.items():
            rendered = rendered.replace("${" + key + "}", str(val))
        return rendered
    if isinstance(value, list):
        return [_render_template(v, context) for v in value]
    if isinstance(value, dict):
        return {k: _render_template(v, context) for k, v in value.items()}
    return value


def _normalize_base_url_to_ws(base_url: str, suffix_path: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.startswith("https://"):
        ws_base = "wss://" + normalized[len("https://") :]
    elif normalized.startswith("http://"):
        ws_base = "ws://" + normalized[len("http://") :]
    elif normalized.startswith("wss://") or normalized.startswith("ws://"):
        ws_base = normalized
    else:
        raise ValueError(f"Invalid base_url: {base_url}")
    parsed = urlparse(ws_base)
    final_path = (parsed.path.rstrip("/") + suffix_path) if parsed.path else suffix_path
    return urlunparse((parsed.scheme, parsed.netloc, final_path, parsed.params, parsed.query, parsed.fragment))


def parse_provider_config_env() -> Dict[str, Any]:
    raw = os.getenv("TRANSCRIBER_PROVIDER_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid TRANSCRIBER_PROVIDER_CONFIG_JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("TRANSCRIBER_PROVIDER_CONFIG_JSON must be a JSON object")
    return parsed


def get_selected_provider() -> str:
    provider = os.getenv("TRANSCRIBER_PROVIDER", "local").strip().lower() or "local"
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported TRANSCRIBER_PROVIDER={provider!r}. Supported values: {supported}")
    return provider


def _get_provider_section(provider_name: str, provider_config: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = provider_config.get(provider_name, {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"TRANSCRIBER_PROVIDER_CONFIG_JSON.{provider_name} must be an object")
    return raw_cfg


def _resolve_api_key(provider_name: str, raw_cfg: Dict[str, Any]) -> str:
    api_key = str(raw_cfg.get("api_key") or "").strip()
    if api_key:
        return api_key
    for env_key in PROVIDER_ENV_KEY_FALLBACKS.get(provider_name, []):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return ""


def _as_int(name: str, value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def resolve_openai_config(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = _get_provider_section("openai", provider_config)
    api_key = _resolve_api_key("openai", raw_cfg)
    if not api_key:
        raise ValueError(
            "OpenAI provider requires api_key in TRANSCRIBER_PROVIDER_CONFIG_JSON.openai.api_key "
            "or REMOTE_TRANSCRIBER_API_KEY"
        )

    model = str(raw_cfg.get("model") or "gpt-4o-mini-transcribe").strip()
    base_url = str(raw_cfg.get("base_url") or "https://api.openai.com/v1").strip().rstrip("/")
    timeout_s = _as_int("openai.timeout_s", raw_cfg.get("timeout_s"), 30)

    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "timeout_s": timeout_s,
    }


def resolve_realtime_provider_config(provider_name: str, provider_config: Dict[str, Any]) -> Dict[str, Any]:
    raw_cfg = _get_provider_section(provider_name, provider_config)
    defaults = PROVIDER_WEBSOCKET_DEFAULTS.get(provider_name, {})
    merged = {
        "ws_url": raw_cfg.get("ws_url", defaults.get("ws_url", "")),
        "model": raw_cfg.get("model", "default"),
        "timeout_s": raw_cfg.get("timeout_s", 30),
        "sample_rate_hz": raw_cfg.get("sample_rate_hz", defaults.get("sample_rate_hz", 16000)),
        "auth_header_name": raw_cfg.get("auth_header_name", defaults.get("auth_header_name", "Authorization")),
        "auth_header_prefix": raw_cfg.get("auth_header_prefix", defaults.get("auth_header_prefix", "Bearer ")),
        "query": raw_cfg.get("query", defaults.get("query", {})),
        "messages": raw_cfg.get("messages", defaults.get("messages", {})),
        "events": raw_cfg.get("events", defaults.get("events", {})),
    }

    api_key = _resolve_api_key(provider_name, raw_cfg)
    if not api_key:
        raise ValueError(
            f"Provider '{provider_name}' requires api_key in TRANSCRIBER_PROVIDER_CONFIG_JSON.{provider_name}.api_key "
            f"or one of env vars: {', '.join(PROVIDER_ENV_KEY_FALLBACKS.get(provider_name, []))}"
        )
    merged["api_key"] = api_key

    ws_url = str(merged["ws_url"] or "").strip()
    if not ws_url:
        raise ValueError(
            f"Provider '{provider_name}' requires ws_url in TRANSCRIBER_PROVIDER_CONFIG_JSON.{provider_name}.ws_url"
        )

    merged["ws_url"] = ws_url
    merged["timeout_s"] = _as_int(f"{provider_name}.timeout_s", merged.get("timeout_s"), 30)
    merged["sample_rate_hz"] = _as_int(f"{provider_name}.sample_rate_hz", merged.get("sample_rate_hz"), 16000)

    if not isinstance(merged["query"], dict):
        raise ValueError(f"{provider_name}.query must be an object")
    if not isinstance(merged["messages"], dict):
        raise ValueError(f"{provider_name}.messages must be an object")
    if not isinstance(merged["events"], dict):
        raise ValueError(f"{provider_name}.events must be an object")

    return merged


def create_provider(
    *,
    selected_provider: str,
    provider_config: Dict[str, Any],
    local_model: Optional[WhisperModel],
    executor,
    beam_size: int,
    best_of: int,
    compression_ratio_threshold: float,
    log_prob_threshold: float,
    no_speech_threshold: float,
    condition_on_previous_text: bool,
    prompt_reset_on_temperature: float,
    vad_filter: bool,
    vad_filter_threshold: float,
    vad_min_silence_duration_ms: int,
    use_temperature_fallback: bool,
    temperature_fallback_chain: List[float],
) -> TranscriptionProvider:
    if selected_provider == "local":
        if local_model is None:
            raise ValueError("Local provider requires a loaded Whisper model")
        return LocalWhisperProvider(
            model=local_model,
            executor=executor,
            beam_size=beam_size,
            best_of=best_of,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            vad_filter=vad_filter,
            vad_filter_threshold=vad_filter_threshold,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            use_temperature_fallback=use_temperature_fallback,
            temperature_fallback_chain=temperature_fallback_chain,
        )

    if selected_provider == "openai":
        cfg = resolve_openai_config(provider_config)
        return OpenAIRealtimeProvider(
            api_key=cfg["api_key"],
            model=cfg["model"],
            base_url=cfg["base_url"],
            timeout_s=cfg["timeout_s"],
        )

    cfg = resolve_realtime_provider_config(selected_provider, provider_config)
    return RealtimeWebsocketProvider(
        provider_name=selected_provider,
        api_key=cfg["api_key"],
        ws_url=cfg["ws_url"],
        model=str(cfg.get("model") or "default"),
        timeout_s=cfg["timeout_s"],
        sample_rate_hz=cfg["sample_rate_hz"],
        auth_header_name=str(cfg.get("auth_header_name") or "Authorization"),
        auth_header_prefix=str(cfg.get("auth_header_prefix") or ""),
        query=cfg.get("query") or {},
        messages=cfg.get("messages") or {},
        events=cfg.get("events") or {},
    )
