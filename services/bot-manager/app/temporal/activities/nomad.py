from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

import httpx
from temporalio import activity

logger = logging.getLogger("bot_manager.temporal.activities.nomad")

NOMAD_AGENT_IP = os.getenv("NOMAD_IP_http")
NOMAD_ADDR = os.getenv("NOMAD_ADDR", f"http://{NOMAD_AGENT_IP}:4646" if NOMAD_AGENT_IP else "").rstrip("/")
BOT_JOB_NAME = os.getenv("VEXA_BOT_JOB_NAME", "vexa-bot")
TRANSCRIPTION_WORKER_JOB_NAME = os.getenv("TRANSCRIPTION_WORKER_JOB_NAME", "vexa-transcription-worker")


class NomadActivityError(RuntimeError):
    pass


def _nomad_required() -> None:
    if not NOMAD_ADDR:
        raise NomadActivityError("NOMAD_ADDR (or NOMAD_IP_http) must be configured for temporal nomad activities")


@activity.defn(name="DispatchBotJobActivity")
async def dispatch_bot_job(payload: Dict[str, Any]) -> Dict[str, str]:
    _nomad_required()
    connection_id = str(payload.get("connection_id") or uuid.uuid4())

    meta = {
        "user_id": str(payload["user_id"]),
        "meeting_id": str(payload["meeting_id"]),
        "meeting_url": payload.get("meeting_url") or "",
        "platform": payload.get("platform") or "",
        "bot_name": payload.get("bot_name") or "",
        "user_token": payload.get("user_token") or "",
        "native_meeting_id": payload.get("native_meeting_id") or "",
        "connection_id": connection_id,
        "language": payload.get("language") or "",
        "task": payload.get("task") or "",
        "transcription_tier": payload.get("transcription_tier") or "realtime",
        "transcribe_enabled": str(payload.get("transcribe_enabled") if payload.get("transcribe_enabled") is not None else True).lower(),
        "recording_enabled": str(payload.get("recording_enabled") if payload.get("recording_enabled") is not None else "").lower(),
        "zoom_obf_token": payload.get("zoom_obf_token") or "",
        "voice_agent_enabled": str(payload.get("voice_agent_enabled") if payload.get("voice_agent_enabled") is not None else "").lower(),
        "default_avatar_url": payload.get("default_avatar_url") or "",
    }

    url = f"{NOMAD_ADDR}/v1/job/{BOT_JOB_NAME}/dispatch"
    logger.info("Dispatching bot job via Nomad for meeting=%s", payload.get("meeting_id"))

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"Meta": meta}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        allocation_or_job_id = data.get("DispatchedJobID") or data.get("EvaluationID")
        if not allocation_or_job_id:
            raise NomadActivityError(f"Nomad dispatch missing DispatchedJobID/EvaluationID: {data}")

    return {
        "container_id": allocation_or_job_id,
        "connection_id": connection_id,
    }


@activity.defn(name="DispatchTranscriptionWorkerActivity")
async def dispatch_transcription_worker(payload: Dict[str, Any]) -> Dict[str, str]:
    _nomad_required()
    tier = str(payload.get("tier") or "realtime")
    worker_id = str(uuid.uuid4())

    meta = {
        "worker_id": worker_id,
        "tier": tier,
        "requested_by": "temporal",
        "meeting_id": str(payload.get("meeting_id") or ""),
        "recording_id": str(payload.get("recording_id") or ""),
        "job_id": str(payload.get("job_id") or ""),
    }

    url = f"{NOMAD_ADDR}/v1/job/{TRANSCRIPTION_WORKER_JOB_NAME}/dispatch"
    logger.info("Dispatching transcription worker via Nomad tier=%s", tier)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"Meta": meta}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        dispatch_id = data.get("DispatchedJobID") or data.get("EvaluationID")
        if not dispatch_id:
            raise NomadActivityError(f"Nomad transcription dispatch missing id: {data}")

    return {
        "dispatch_id": dispatch_id,
        "worker_id": worker_id,
    }


@activity.defn(name="StopNomadAllocationActivity")
async def stop_nomad_allocation(payload: Dict[str, Any]) -> bool:
    _nomad_required()
    allocation_id = str(payload.get("allocation_id") or payload.get("container_id") or "").strip()
    if not allocation_id:
        return True

    stop_url = f"{NOMAD_ADDR}/v1/allocation/{allocation_id}/stop"
    deregister_url = f"{NOMAD_ADDR}/v1/job/{allocation_id}/deregister?purge=true"

    async with httpx.AsyncClient() as client:
        stop_resp = await client.post(stop_url, timeout=10)
        if stop_resp.status_code in (200, 202, 404):
            return True

        # fallback if id is a dispatched job id, not allocation id
        deregister_resp = await client.post(deregister_url, timeout=10)
        return deregister_resp.status_code in (200, 202, 404)


@activity.defn(name="CheckNomadAllocationActivity")
async def check_nomad_allocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    _nomad_required()
    allocation_id = str(payload.get("allocation_id") or payload.get("container_id") or "").strip()
    if not allocation_id:
        return {"running": False, "client_status": "unknown"}

    url = f"{NOMAD_ADDR}/v1/allocation/{allocation_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        if resp.status_code == 404:
            return {"running": False, "client_status": "not_found"}
        resp.raise_for_status()
        data = resp.json()
        status = data.get("ClientStatus", "unknown")
        return {"running": status in ("running", "pending"), "client_status": status, "raw": data}
