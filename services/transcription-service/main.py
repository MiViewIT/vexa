"""
Vexa-Compatible Transcription Service (PoC)
Implements OpenAI Whisper API format for seamless integration with Vexa
"""
import os
import io
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import uvicorn
from faster_whisper import WhisperModel

from transcription_provider import (
    ProviderRequest,
    TranscriptionProvider,
    create_provider,
    get_selected_provider,
    parse_provider_config_env,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WORKER_ID = os.getenv("WORKER_ID", "1")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3-turbo")

# Device detection: Use environment variable or default to cuda for GPU containers
# CTranslate2 (used by faster-whisper) will automatically detect and use CUDA if available
DEVICE = os.getenv("DEVICE", "cuda")

# Compute type optimization: Use INT8 for optimal VRAM efficiency
COMPUTE_TYPE_ENV = os.getenv("COMPUTE_TYPE", "").strip().lower()
if COMPUTE_TYPE_ENV:
    COMPUTE_TYPE = COMPUTE_TYPE_ENV
else:
    COMPUTE_TYPE = "int8"

# CPU threads configuration (for CPU mode optimization)
CPU_THREADS = int(os.getenv("CPU_THREADS", "0"))  # 0 = auto-detect


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, None)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, None)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid int env {name}={raw!r}, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, None)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float env {name}={raw!r}, using default {default}")
        return default


# WhisperLive-inspired defaults (can be overridden via env)
BEAM_SIZE = _env_int("BEAM_SIZE", 5)
BEST_OF = _env_int("BEST_OF", 5)
COMPRESSION_RATIO_THRESHOLD = _env_float("COMPRESSION_RATIO_THRESHOLD", 2.4)
LOG_PROB_THRESHOLD = _env_float("LOG_PROB_THRESHOLD", -1.0)
NO_SPEECH_THRESHOLD = _env_float("NO_SPEECH_THRESHOLD", 0.6)
CONDITION_ON_PREVIOUS_TEXT = _env_bool("CONDITION_ON_PREVIOUS_TEXT", True)
PROMPT_RESET_ON_TEMPERATURE = _env_float("PROMPT_RESET_ON_TEMPERATURE", 0.5)

# VAD parameters
VAD_FILTER = _env_bool("VAD_FILTER", True)
VAD_FILTER_THRESHOLD = _env_float("VAD_FILTER_THRESHOLD", 0.5)
VAD_MIN_SILENCE_DURATION_MS = _env_int("VAD_MIN_SILENCE_DURATION_MS", 160)

# Temperature fallback chain
USE_TEMPERATURE_FALLBACK = _env_bool("USE_TEMPERATURE_FALLBACK", False)
TEMPERATURE_FALLBACK_CHAIN = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# API Token Authentication
API_TOKEN = os.getenv("API_TOKEN", "").strip()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Provider selection/config
try:
    SELECTED_PROVIDER = get_selected_provider()
    PROVIDER_CONFIG = parse_provider_config_env()
except ValueError as exc:
    # Fail fast before service accepts traffic.
    raise RuntimeError(str(exc)) from exc


async def verify_api_token(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER)
) -> bool:
    """Verify API token - supports both X-API-Key and Authorization Bearer"""
    if not API_TOKEN:
        # If no token configured, allow all requests (backward compatibility)
        logger.warning("API_TOKEN not configured - allowing all requests")
        return True

    # Try X-API-Key header first
    if api_key and api_key == API_TOKEN:
        return True

    # Try Authorization Bearer header (for compatibility)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "").strip()
        if token == API_TOKEN:
            return True

    logger.warning(f"Invalid or missing API token - X-API-Key: {api_key is not None}, Authorization: {bool(auth_header)}")
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API token"
    )


app = FastAPI(
    title="Vexa Transcription Service",
    description="OpenAI Whisper API compatible transcription service",
    version="1.0.0"
)

# Global instances
model: Optional[WhisperModel] = None
transcription_provider: Optional[TranscriptionProvider] = None
provider_ready = False
provider_error: Optional[str] = None

# Load management: Global concurrency limit and bounded queue
MAX_CONCURRENT_TRANSCRIPTIONS = _env_int("MAX_ACTIVE_REQUESTS", _env_int("MAX_CONCURRENT_TRANSCRIPTIONS", 2))
MAX_QUEUE_SIZE = _env_int("MAX_QUEUE_SIZE", 10)  # Max requests waiting in queue

# Backpressure strategy:
FAIL_FAST_WHEN_BUSY = _env_bool("FAIL_FAST_WHEN_BUSY", True)
BUSY_RETRY_AFTER_S = _env_int("BUSY_RETRY_AFTER_S", 1)
REALTIME_RESERVED_SLOTS = _env_int("REALTIME_RESERVED_SLOTS", 1)

# Semaphore to limit concurrent transcriptions (protects GPU/CPU from overload)
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# Thread pool for running blocking transcription calls
transcription_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)

# Queue to track waiting requests
waiting_requests = 0
waiting_requests_lock = asyncio.Lock()

# Active in-flight counters per tier for admission decisions.
active_realtime_requests = 0
active_deferred_requests = 0
active_requests_lock = asyncio.Lock()


def _normalize_transcription_tier(raw: Optional[str]) -> str:
    tier = (raw or "realtime").strip().lower()
    return tier if tier in ("realtime", "deferred") else "realtime"


def _deferred_capacity_available(active_rt: int, active_df: int) -> bool:
    deferred_limit = max(0, MAX_CONCURRENT_TRANSCRIPTIONS - REALTIME_RESERVED_SLOTS)
    total_active = active_rt + active_df
    return deferred_limit > 0 and active_df < deferred_limit and total_active < MAX_CONCURRENT_TRANSCRIPTIONS


@app.on_event("startup")
async def startup_event():
    """Initialize transcription provider on startup"""
    global model, transcription_provider, provider_ready, provider_error

    logger.info(f"Worker {WORKER_ID} starting up...")
    logger.info(f"Provider: {SELECTED_PROVIDER}, Device: {DEVICE}, Model: {MODEL_SIZE}, Compute: {COMPUTE_TYPE}")
    logger.info(
        "Quality params - "
        f"beam_size={BEAM_SIZE}, best_of={BEST_OF}, "
        f"cond_prev_text={CONDITION_ON_PREVIOUS_TEXT}, "
        f"compression_ratio_threshold={COMPRESSION_RATIO_THRESHOLD}, "
        f"log_prob_threshold={LOG_PROB_THRESHOLD}, "
        f"no_speech_threshold={NO_SPEECH_THRESHOLD}, "
        f"vad_filter={VAD_FILTER}"
    )

    try:
        if SELECTED_PROVIDER == "local":
            model_kwargs = {
                "model_size_or_path": MODEL_SIZE,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "download_root": "/app/models"
            }

            if DEVICE == "cpu" and CPU_THREADS > 0:
                model_kwargs["cpu_threads"] = CPU_THREADS
                logger.info(f"Worker {WORKER_ID} using {CPU_THREADS} CPU threads")

            model = WhisperModel(**model_kwargs)
            logger.info(f"Worker {WORKER_ID} ready - Local Whisper model loaded successfully")

        transcription_provider = create_provider(
            selected_provider=SELECTED_PROVIDER,
            provider_config=PROVIDER_CONFIG,
            local_model=model,
            executor=transcription_executor,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            prompt_reset_on_temperature=PROMPT_RESET_ON_TEMPERATURE,
            vad_filter=VAD_FILTER,
            vad_filter_threshold=VAD_FILTER_THRESHOLD,
            vad_min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            use_temperature_fallback=USE_TEMPERATURE_FALLBACK,
            temperature_fallback_chain=TEMPERATURE_FALLBACK_CHAIN,
        )
        provider_ready = True
        provider_error = None
        logger.info(f"Worker {WORKER_ID} provider initialized successfully: {SELECTED_PROVIDER}")
    except Exception as exc:
        provider_ready = False
        provider_error = str(exc)
        logger.error(f"Failed to initialize provider '{SELECTED_PROVIDER}': {exc}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer"""
    health_status = {
        "status": "healthy" if provider_ready else "unhealthy",
        "worker_id": WORKER_ID,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": SELECTED_PROVIDER,
        "model": MODEL_SIZE if SELECTED_PROVIDER == "local" else None,
        "device": DEVICE,
        "gpu_available": DEVICE == "cuda",
    }

    if SELECTED_PROVIDER == "local" and DEVICE == "cuda":
        health_status["compute_type"] = COMPUTE_TYPE
    if provider_error:
        health_status["error"] = provider_error

    if not provider_ready:
        return JSONResponse(content=health_status, status_code=503)

    return health_status


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    requested_model: str = Form(..., alias="model"),
    temperature: str = Form("0"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("verbose_json"),
    timestamp_granularities: str = Form("segment"),
    transcription_tier_form: Optional[str] = Form(None, alias="transcription_tier"),
    task: str = Form("transcribe"),
    _: bool = Depends(verify_api_token)
):
    """
    OpenAI Whisper API compatible transcription endpoint

    Required by Vexa's RemoteTranscriber:
    - Accepts multipart/form-data with audio file
    - Returns verbose_json format with segments
    - Includes timing, language, and segment details

    Load management:
    - Limits concurrent transcriptions to prevent GPU/CPU overload
    - Returns 429/503 when queue is full to signal backpressure
    """
    if not requested_model:
        raise HTTPException(status_code=400, detail="Model parameter is required")
    if transcription_provider is None:
        raise HTTPException(status_code=503, detail="Transcription provider not initialized")

    global waiting_requests, active_realtime_requests, active_deferred_requests

    tier_from_header = request.headers.get("X-Transcription-Tier")
    transcription_tier = _normalize_transcription_tier(transcription_tier_form or tier_from_header)

    semaphore_acquired = False
    waiting_counted = False
    active_counted = False

    # Load management: Check queue size before accepting request
    async with waiting_requests_lock:
        async with active_requests_lock:
            current_active_rt = active_realtime_requests
            current_active_df = active_deferred_requests

        if transcription_tier == "deferred":
            if not _deferred_capacity_available(current_active_rt, current_active_df):
                raise HTTPException(
                    status_code=503,
                    detail="Deferred tier is out of capacity. Please retry later.",
                    headers={"Retry-After": str(max(1, BUSY_RETRY_AFTER_S))},
                )

        if FAIL_FAST_WHEN_BUSY and (transcription_semaphore.locked() or waiting_requests > 0):
            raise HTTPException(
                status_code=503,
                detail="Service busy. Please retry later.",
                headers={"Retry-After": str(max(1, BUSY_RETRY_AFTER_S))},
            )

        if waiting_requests >= MAX_QUEUE_SIZE:
            logger.warning(
                f"Worker {WORKER_ID} queue full ({waiting_requests}/{MAX_QUEUE_SIZE}). "
                f"Rejecting request with 503."
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily overloaded. Please retry later.",
                headers={"Retry-After": str(max(1, BUSY_RETRY_AFTER_S))}
            )
        waiting_requests += 1
        waiting_counted = True

    try:
        await transcription_semaphore.acquire()
        semaphore_acquired = True

        async with waiting_requests_lock:
            if waiting_counted:
                waiting_requests -= 1
                waiting_counted = False

        async with active_requests_lock:
            if transcription_tier == "deferred":
                active_deferred_requests += 1
            else:
                active_realtime_requests += 1
            active_counted = True

        start_time = time.time()
        logger.info(
            f"Worker {WORKER_ID} received transcription request - "
            f"provider={SELECTED_PROVIDER}, tier={transcription_tier}, "
            f"filename={file.filename}, content_type={file.content_type}"
        )

        audio_bytes = await file.read()
        logger.info(f"Worker {WORKER_ID} read {len(audio_bytes)} bytes of audio data")

        audio_io = io.BytesIO(audio_bytes)
        try:
            audio_array, sample_rate = sf.read(audio_io, dtype=np.float32)
            logger.info(f"Worker {WORKER_ID} decoded audio - shape: {audio_array.shape}, sample_rate: {sample_rate}")
        except Exception as exc:
            logger.error(f"Worker {WORKER_ID} failed to decode audio with soundfile: {exc}")
            raise HTTPException(status_code=400, detail=f"Failed to decode audio file: {exc}")

        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
            logger.info(f"Worker {WORKER_ID} converted to mono - shape: {audio_array.shape}")

        audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

        provider_request = ProviderRequest(
            language=language,
            prompt=prompt,
            task=task,
            transcription_tier=transcription_tier,
            requested_model=requested_model,
            temperature=float(temperature) if temperature else 0.0,
            sample_rate=int(sample_rate),
        )

        result = await transcription_provider.transcribe(audio_array, provider_request)

        processing_time = time.time() - start_time
        logger.info(
            f"Worker {WORKER_ID} completed in {processing_time:.2f}s - "
            f"provider={SELECTED_PROVIDER}, duration={result.duration:.2f}s, "
            f"segments={len(result.segments)}, language={result.language}"
        )

        response = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
        }
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Worker {WORKER_ID} transcription failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if active_counted:
            async with active_requests_lock:
                if transcription_tier == "deferred":
                    active_deferred_requests = max(0, active_deferred_requests - 1)
                else:
                    active_realtime_requests = max(0, active_realtime_requests - 1)
            active_counted = False

        if waiting_counted:
            async with waiting_requests_lock:
                waiting_requests = max(0, waiting_requests - 1)
            waiting_counted = False

        if semaphore_acquired:
            transcription_semaphore.release()


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Vexa Transcription Service",
        "worker_id": WORKER_ID,
        "provider": SELECTED_PROVIDER,
        "model": MODEL_SIZE if SELECTED_PROVIDER == "local" else None,
        "device": DEVICE,
        "status": "ready" if provider_ready else "initializing",
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
