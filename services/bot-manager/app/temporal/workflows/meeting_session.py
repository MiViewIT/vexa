from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn
class MeetingSessionWorkflow:
    """Durable meeting lifecycle orchestration.

    Data path remains direct bot->transcription endpoint; this workflow orchestrates
    job dispatch, timeout handling, status signals, and cleanup.
    """

    def __init__(self) -> None:
        self._state: str = "requested"
        self._stop_requested: bool = False
        self._container_id: Optional[str] = None
        self._connection_id: Optional[str] = None
        self._last_status_payload: Optional[Dict[str, Any]] = None
        self._last_reconfigure: Optional[Dict[str, Any]] = None

    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=5,
        )

        dispatch = await workflow.execute_activity(
            "DispatchBotJobActivity",
            {
                **payload,
                "connection_id": payload.get("session_uid") or payload.get("connection_id"),
            },
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        self._container_id = dispatch.get("container_id")
        self._connection_id = dispatch.get("connection_id")

        # Kick/ensure transcription capacity per tier.
        await workflow.execute_activity(
            "DispatchTranscriptionWorkerActivity",
            {
                "tier": payload.get("transcription_tier") or "realtime",
                "meeting_id": payload.get("meeting_id"),
            },
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Wait for signals / status transitions with timeout guards.
        join_timeout = int(payload.get("join_timeout_s") or 180)
        active_timeout = int(payload.get("active_heartbeat_timeout_s") or 1800)

        try:
            await workflow.wait_condition(
                lambda: self._state in ("joining", "awaiting_admission", "active", "completed", "failed") or self._stop_requested,
                timeout=timedelta(seconds=join_timeout),
            )
        except TimeoutError:
            self._state = "failed"

        if self._stop_requested:
            self._state = "stopping"

        if self._state not in ("completed", "failed") and not self._stop_requested:
            try:
                await workflow.wait_condition(
                    lambda: self._state in ("active", "completed", "failed") or self._stop_requested,
                    timeout=timedelta(seconds=active_timeout),
                )
            except TimeoutError:
                self._state = "failed"

        final_state = self._state
        if self._stop_requested and final_state not in ("completed", "failed"):
            final_state = "completed"

        if final_state not in ("completed", "failed"):
            final_state = "failed"

        await workflow.execute_activity(
            "FinalizeSessionActivity",
            {
                "meeting_id": payload["meeting_id"],
                "status": final_state,
                "reason": "workflow_finalize",
            },
            schedule_to_close_timeout=timedelta(seconds=20),
        )

        if self._container_id:
            await workflow.execute_activity(
                "StopNomadAllocationActivity",
                {"container_id": self._container_id},
                schedule_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        return {
            "state": final_state,
            "container_id": self._container_id,
            "connection_id": self._connection_id,
            "stop_requested": self._stop_requested,
        }

    @workflow.signal
    def StopSession(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self._stop_requested = True
        if self._state not in ("completed", "failed"):
            self._state = "stopping"

    @workflow.signal
    def ReconfigureSession(self, payload: Dict[str, Any]) -> None:
        self._last_reconfigure = payload

    @workflow.signal
    def BotStatusUpdate(self, payload: Dict[str, Any]) -> None:
        status = str(payload.get("status") or "").strip().lower()
        if not status:
            return
        self._last_status_payload = payload
        if status in ("joining", "awaiting_admission", "active", "completed", "failed", "stopping"):
            self._state = status

    @workflow.query
    def GetSessionState(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "container_id": self._container_id,
            "connection_id": self._connection_id,
            "stop_requested": self._stop_requested,
            "last_status_payload": self._last_status_payload,
            "last_reconfigure": self._last_reconfigure,
        }

    @workflow.query
    def GetCurrentBotAllocation(self) -> Dict[str, Any]:
        return {
            "container_id": self._container_id,
            "connection_id": self._connection_id,
        }
