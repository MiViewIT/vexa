from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from app.temporal.activities import (
    dispatch_bot_job,
    dispatch_transcription_worker,
    stop_nomad_allocation,
    check_nomad_allocation,
    finalize_session,
    mark_job_completed,
    run_deferred_transcription,
    persist_segments,
)
from app.temporal.workflows import (
    MeetingSessionWorkflow,
    DeferredTranscriptionWorkflow,
    TranscriptionCapacityWorkflow,
)

logger = logging.getLogger("bot_manager.temporal.worker")


async def run_worker() -> None:
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queues = [
        os.getenv("TEMPORAL_TASK_QUEUE_MEETING", "vexa-meeting"),
        os.getenv("TEMPORAL_TASK_QUEUE_DEFERRED", "vexa-deferred"),
        os.getenv("TEMPORAL_TASK_QUEUE_CAPACITY", "vexa-capacity"),
    ]

    client = await Client.connect(address, namespace=namespace)

    workers = []
    for tq in task_queues:
        workers.append(
            Worker(
                client,
                task_queue=tq,
                workflows=[
                    MeetingSessionWorkflow,
                    DeferredTranscriptionWorkflow,
                    TranscriptionCapacityWorkflow,
                ],
                activities=[
                    dispatch_bot_job,
                    dispatch_transcription_worker,
                    stop_nomad_allocation,
                    check_nomad_allocation,
                    finalize_session,
                    mark_job_completed,
                    run_deferred_transcription,
                    persist_segments,
                ],
            )
        )

    logger.info("Starting Temporal workers for task queues: %s", task_queues)
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    asyncio.run(run_worker())
