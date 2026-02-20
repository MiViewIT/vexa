from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow


@workflow.defn
class TranscriptionCapacityWorkflow:
    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tier = str(payload.get("tier") or "realtime")
        target_workers = int(payload.get("target_workers") or 1)
        interval_s = int(payload.get("reconcile_interval_s") or 30)
        loops = int(payload.get("max_loops") or 1)

        dispatched = 0
        for _ in range(max(1, loops)):
            for _ in range(max(0, target_workers)):
                await workflow.execute_activity(
                    "DispatchTranscriptionWorkerActivity",
                    {"tier": tier},
                    schedule_to_close_timeout=timedelta(seconds=20),
                )
                dispatched += 1
            await workflow.sleep(timedelta(seconds=interval_s))

        return {
            "tier": tier,
            "dispatched": dispatched,
            "target_workers": target_workers,
        }
