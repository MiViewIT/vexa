from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from temporalio import activity
from sqlalchemy import select

from shared_models.database import async_session_local
from shared_models.models import Meeting, TranscriptionJob
from shared_models.schemas import MeetingStatus

logger = logging.getLogger("bot_manager.temporal.activities.state")


@activity.defn(name="FinalizeSessionActivity")
async def finalize_session(payload: Dict[str, Any]) -> bool:
    meeting_id = int(payload["meeting_id"])
    status = str(payload.get("status") or MeetingStatus.COMPLETED.value)
    reason = payload.get("reason")

    async with async_session_local() as db:
        meeting = await db.get(Meeting, meeting_id)
        if not meeting:
            logger.warning("FinalizeSessionActivity meeting not found id=%s", meeting_id)
            return False

        if status in (MeetingStatus.COMPLETED.value, MeetingStatus.FAILED.value):
            meeting.status = status
        if meeting.end_time is None:
            meeting.end_time = datetime.utcnow()

        data = dict(meeting.data or {})
        temporal = dict(data.get("temporal") or {})
        temporal["last_finalize_reason"] = reason
        temporal["last_finalize_ts"] = datetime.utcnow().isoformat()
        data["temporal"] = temporal
        meeting.data = data

        await db.commit()
        return True


@activity.defn(name="MarkJobCompletedActivity")
async def mark_job_completed(payload: Dict[str, Any]) -> bool:
    job_id = int(payload["job_id"])
    status = str(payload.get("status") or "completed")
    error_message: Optional[str] = payload.get("error_message")
    progress = payload.get("progress")

    async with async_session_local() as db:
        job = await db.get(TranscriptionJob, job_id)
        if not job:
            logger.warning("MarkJobCompletedActivity transcription job not found id=%s", job_id)
            return False

        job.status = status
        if status == "processing" and job.started_at is None:
            job.started_at = datetime.utcnow()
        if status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()

        if error_message:
            job.error_message = error_message
        if progress is not None:
            try:
                job.progress = float(progress)
            except Exception:
                pass

        await db.commit()
        return True


@activity.defn(name="RunDeferredTranscriptionActivity")
async def run_deferred_transcription(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Phase 1 placeholder: orchestration wires job lifecycle and capacity dispatch.
    # Actual batch decode + segmentation can be plugged in here next.
    job_id = int(payload["job_id"])
    logger.info("RunDeferredTranscriptionActivity placeholder executed for job_id=%s", job_id)
    return {
        "job_id": job_id,
        "segments_count": 0,
        "status": "completed",
    }


@activity.defn(name="PersistSegmentsActivity")
async def persist_segments(payload: Dict[str, Any]) -> bool:
    # Phase 1 placeholder to preserve workflow contract.
    logger.info("PersistSegmentsActivity placeholder called payload_keys=%s", list(payload.keys()))
    return True
