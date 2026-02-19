import asyncio
import json

import numpy as np
import pytest

import transcription_provider as tp


def test_parse_provider_config_env_valid(monkeypatch):
    cfg = {"openai": {"model": "gpt-4o-mini-transcribe"}}
    monkeypatch.setenv("TRANSCRIBER_PROVIDER_CONFIG_JSON", json.dumps(cfg))
    parsed = tp.parse_provider_config_env()
    assert parsed == cfg


def test_parse_provider_config_env_invalid(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_PROVIDER_CONFIG_JSON", "{invalid-json")
    with pytest.raises(ValueError, match="Invalid TRANSCRIBER_PROVIDER_CONFIG_JSON"):
        tp.parse_provider_config_env()


def test_openai_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("REMOTE_TRANSCRIBER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires api_key"):
        tp.resolve_openai_config({"openai": {"model": "gpt-4o-mini-transcribe"}})


def test_get_selected_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_PROVIDER", "nonsense")
    with pytest.raises(ValueError, match="Unsupported TRANSCRIBER_PROVIDER"):
        tp.get_selected_provider()


def test_realtime_provider_requires_ws_url(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    with pytest.raises(ValueError, match="requires ws_url"):
        tp.resolve_realtime_provider_config("deepgram", {"deepgram": {}})


def test_openai_provider_normalization_single_segment(monkeypatch):
    provider = tp.OpenAIRealtimeProvider(
        api_key="test-key",
        model="gpt-4o-mini-transcribe",
        base_url="https://api.openai.com/v1",
        timeout_s=5,
    )

    async def fake_ws(audio_array, request):
        return "hello world", "en"

    monkeypatch.setattr(provider, "_transcribe_via_ws", fake_ws)

    req = tp.ProviderRequest(
        language=None,
        prompt=None,
        task="transcribe",
        transcription_tier="realtime",
        requested_model="whisper-1",
        temperature=0.0,
        sample_rate=16000,
    )
    audio = np.zeros(16000, dtype=np.float32)

    result = asyncio.run(provider.transcribe(audio, req))
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration == pytest.approx(1.0)
    assert len(result.segments) == 1
    assert result.segments[0]["start"] == 0.0
    assert result.segments[0]["end"] == pytest.approx(1.0)


def test_deepgram_provider_builds_from_registry(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    cfg = {
        "deepgram": {
            "ws_url": "wss://api.deepgram.com/v1/listen",
        }
    }
    provider = tp.create_provider(
        selected_provider="deepgram",
        provider_config=cfg,
        local_model=None,
        executor=None,
        beam_size=5,
        best_of=5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        vad_filter=True,
        vad_filter_threshold=0.5,
        vad_min_silence_duration_ms=160,
        use_temperature_fallback=False,
        temperature_fallback_chain=[0.0],
    )
    assert isinstance(provider, tp.RealtimeWebsocketProvider)
