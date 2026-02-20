from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn
class DeferredTranscriptionWorkflow:
    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=5,
        )

        job_id = int(payload["job_id"])

        await workflow.execute_activity(
            "MarkJobCompletedActivity",
            {"job_id": job_id, "status": "processing", "progress": 0.05},
            schedule_to_close_timeout=timedelta(seconds=20),
        )

        try:
            await workflow.execute_activity(
                "DispatchTranscriptionWorkerActivity",
                {
                    "tier": "deferred",
                    "meeting_id": payload.get("meeting_id"),
                    "recording_id": payload.get("recording_id"),
                    "job_id": job_id,
                },
                schedule_to_close_timeout=timedelta(seconds=30),
                retry_policy=retry_policy,
            )

            result = await workflow.execute_activity(
                "RunDeferredTranscriptionActivity",
                payload,
                schedule_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy,
            )

            await workflow.execute_activity(
                "PersistSegmentsActivity",
                {
                    "job_id": job_id,
                    "segments_count": result.get("segments_count", 0),
                    "payload": result,
                },
                schedule_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            await workflow.execute_activity(
                "MarkJobCompletedActivity",
                {
                    "job_id": job_id,
                    "status": "completed",
                    "progress": 1.0,
                },
                schedule_to_close_timeout=timedelta(seconds=20),
            )

            return {
                "job_id": job_id,
                "status": "completed",
                "segments_count": result.get("segments_count", 0),
            }
        except Exception as exc:
            await workflow.execute_activity(
                "MarkJobCompletedActivity",
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error_message": str(exc),
                },
                schedule_to_close_timeout=timedelta(seconds=20),
            )
            raise
