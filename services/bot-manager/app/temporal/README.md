# Temporal orchestration package

This package contains the Temporal control-plane implementation for `bot-manager`.

## Layout

- `workflows/meeting_session.py`: durable lifecycle for one meeting session.
- `workflows/deferred_transcription.py`: deferred/batch transcription workflow.
- `workflows/transcription_capacity.py`: simple capacity loop for transcription workers.
- `activities/nomad.py`: Nomad dispatch/stop/check activities for ephemeral workers.
- `activities/state.py`: DB state update activities.
- `worker.py`: Temporal worker process entrypoint.

## Design boundaries

- Realtime audio data does not flow through Temporal.
- Temporal coordinates bot/transcription worker lifecycle, retries, timers, and deferred tasks.
- Nomad remains the data-plane provisioner for ephemeral workers.

## Adding new activities/workflows

1. Add activity in `activities/` and register it in `worker.py`.
2. Add workflow in `workflows/` and register it in `worker.py`.
3. Route API/orchestrator calls through `app/orchestrators/temporal.py`.
4. Keep payloads JSON-serializable and idempotent where possible.
