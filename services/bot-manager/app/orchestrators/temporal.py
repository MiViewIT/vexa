"""Temporal orchestrator backend.

This backend uses Temporal as the durable control plane and Nomad activities
for ephemeral worker provisioning.

It preserves the bot-manager orchestrator API surface, so existing endpoints
continue to call start/stop/status helpers while orchestration is durable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy

from app.temporal.types import MeetingWorkflowInput
from app.temporal.workflows.meeting_session import MeetingSessionWorkflow
from app.temporal.workflows.deferred_transcription import DeferredTranscriptionWorkflow

logger = logging.getLogger("bot_manager.temporal_orchestrator")

TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")
TEMPORAL_TLS_ENABLED = os.getenv("TEMPORAL_TLS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
TEMPORAL_TASK_QUEUE_MEETING = os.getenv("TEMPORAL_TASK_QUEUE_MEETING", "vexa-meeting")
TEMPORAL_TASK_QUEUE_DEFERRED = os.getenv("TEMPORAL_TASK_QUEUE_DEFERRED", "vexa-deferred")
TEMPORAL_TASK_QUEUE_CAPACITY = os.getenv("TEMPORAL_TASK_QUEUE_CAPACITY", "vexa-capacity")
TEMPORAL_WORKFLOW_ID_PREFIX = os.getenv("TEMPORAL_WORKFLOW_ID_PREFIX", "")

_client: Optional[Client] = None
_client_lock = asyncio.Lock()


def _wf_id(prefix: str) -> str:
    if TEMPORAL_WORKFLOW_ID_PREFIX:
        return f"{TEMPORAL_WORKFLOW_ID_PREFIX}:{prefix}"
    return prefix


async def _get_temporal_client() -> Client:
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client
        _client = await Client.connect(TEMPORAL_ADDRESS, namespace=TEMPORAL_NAMESPACE, tls=TEMPORAL_TLS_ENABLED)
        logger.info("Temporal client connected addr=%s namespace=%s", TEMPORAL_ADDRESS, TEMPORAL_NAMESPACE)
        return _client


async def check_orchestrator_health() -> Dict[str, Any]:
    try:
        await _get_temporal_client()
        return {
            "ok": True,
            "address": TEMPORAL_ADDRESS,
            "namespace": TEMPORAL_NAMESPACE,
            "health": "connected",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Temporal orchestrator health check failed: %s", e)
        return {"ok": False, "address": TEMPORAL_ADDRESS, "namespace": TEMPORAL_NAMESPACE, "error": str(e)}


def get_socket_session(*_args, **_kwargs):  # compatibility
    return None


def close_client() -> None:
    # temporalio python client has no explicit close API.
    logger.info("Temporal orchestrator close_client called")


close_docker_client = close_client


async def start_meeting_workflow(workflow_input: Dict[str, Any], session_uid: Optional[str] = None) -> Tuple[str, str, str]:
    client = await _get_temporal_client()
    session_uid = session_uid or str(uuid.uuid4())
    workflow_id = _wf_id(f"meeting:{workflow_input['meeting_id']}:session:{session_uid}")

    handle = await client.start_workflow(
        MeetingSessionWorkflow.run,
        {
            **workflow_input,
            "session_uid": session_uid,
        },
        id=workflow_id,
        task_queue=TEMPORAL_TASK_QUEUE_MEETING,
        execution_timeout=timedelta(hours=6),
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
    )
    logger.info("Started MeetingSessionWorkflow workflow_id=%s run_id=%s", workflow_id, handle.result_run_id)
    return workflow_id, handle.result_run_id, session_uid


async def signal_stop_meeting(workflow_id: str, reason: Optional[str] = None) -> None:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("StopSession", {"reason": reason or "stop_requested"})


async def signal_reconfigure_meeting(
    workflow_id: str,
    *,
    language: Optional[str] = None,
    task: Optional[str] = None,
    transcription_tier: Optional[str] = None,
) -> None:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(
        "ReconfigureSession",
        {
            "language": language,
            "task": task,
            "transcription_tier": transcription_tier,
        },
    )


async def signal_bot_status_update(
    workflow_id: str,
    *,
    status: str,
    reason: Optional[str] = None,
    exit_code: Optional[int] = None,
    container_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(
        "BotStatusUpdate",
        {
            "status": status,
            "reason": reason,
            "exit_code": exit_code,
            "container_id": container_id,
            "payload": payload or {},
        },
    )


async def query_meeting_workflow(workflow_id: str) -> Dict[str, Any]:
    client = await _get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    state = await handle.query("GetSessionState")
    return state


async def start_deferred_transcription_workflow(payload: Dict[str, Any]) -> Tuple[str, str]:
    client = await _get_temporal_client()
    workflow_id = _wf_id(f"deferred-tx:{payload['recording_id']}:{payload['job_id']}")
    handle = await client.start_workflow(
        DeferredTranscriptionWorkflow.run,
        payload,
        id=workflow_id,
        task_queue=TEMPORAL_TASK_QUEUE_DEFERRED,
        execution_timeout=timedelta(hours=6),
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
    )
    return workflow_id, handle.result_run_id


async def start_bot_container(
    user_id: int,
    meeting_id: int,
    meeting_url: Optional[str],
    platform: str,
    bot_name: Optional[str],
    user_token: str,
    native_meeting_id: str,
    language: Optional[str],
    task: Optional[str],
    transcription_tier: Optional[str] = "realtime",
    recording_enabled: Optional[bool] = None,
    transcribe_enabled: Optional[bool] = None,
    zoom_obf_token: Optional[str] = None,
    voice_agent_enabled: Optional[bool] = None,
    default_avatar_url: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Compatibility adapter: starts a meeting workflow.

    Returns:
      (workflow_id, connection_id)
    """
    wf_input = MeetingWorkflowInput(
        meeting_id=meeting_id,
        user_id=user_id,
        meeting_url=meeting_url,
        platform=platform,
        bot_name=bot_name,
        user_token=user_token,
        native_meeting_id=native_meeting_id,
        language=language,
        task=task,
        transcription_tier=transcription_tier or "realtime",
        transcribe_enabled=transcribe_enabled,
        recording_enabled=recording_enabled,
        zoom_obf_token=zoom_obf_token,
        voice_agent_enabled=voice_agent_enabled,
        default_avatar_url=default_avatar_url,
    )

    workflow_id, run_id, session_uid = await start_meeting_workflow(wf_input.to_payload())
    logger.info("Temporal start_bot_container workflow_id=%s run_id=%s session_uid=%s", workflow_id, run_id, session_uid)
    return workflow_id, session_uid


def stop_bot_container(container_id: str) -> bool:
    """Compatibility adapter: signal workflow stop by workflow id.

    Existing callsites treat container_id as opaque; in temporal mode we store
    workflow_id in this field.
    """
    try:
        asyncio.get_running_loop()
        # If we are already on an event loop, schedule fire-and-forget and return.
        asyncio.create_task(signal_stop_meeting(container_id, reason="stop_bot_container"))
        return True
    except RuntimeError:
        try:
            asyncio.run(signal_stop_meeting(container_id, reason="stop_bot_container"))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to stop temporal workflow %s: %s", container_id, e, exc_info=True)
            return False


async def get_running_bots_status(user_id: int) -> List[Dict[str, Any]]:
    # Keep API compatibility with /bots/status without requiring Temporal visibility indexing.
    from shared_models.database import async_session_local
    from shared_models.models import Meeting
    from sqlalchemy import select, desc

    async with async_session_local() as db:
        result = await db.execute(
            select(Meeting)
            .where(
                Meeting.user_id == user_id,
                Meeting.status.in_(["requested", "joining", "awaiting_admission", "active", "stopping"]),
            )
            .order_by(desc(Meeting.created_at))
        )
        meetings = result.scalars().all()

    statuses = []
    for m in meetings:
        data = m.data or {}
        temporal = data.get("temporal") if isinstance(data, dict) else {}
        statuses.append(
            {
                "container_id": m.bot_container_id,
                "container_name": temporal.get("workflow_id") or m.bot_container_id,
                "platform": m.platform,
                "native_meeting_id": m.platform_specific_id,
                "status": m.status,
                "normalized_status": "Up" if m.status == "active" else "Starting",
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "labels": {
                    "meeting_id": str(m.id),
                    "user_id": str(m.user_id),
                    "workflow_id": temporal.get("workflow_id"),
                    "run_id": temporal.get("run_id"),
                },
                "meeting_id_from_name": str(m.id),
            }
        )
    return statuses


async def verify_container_running(container_id: str) -> bool:
    try:
        state = await query_meeting_workflow(container_id)
        return state.get("state") not in ("completed", "failed")
    except Exception:
        return False


# Shared session recorder import for compatibility with existing flow
from app.orchestrator_utils import _record_session_start  # noqa: E402
