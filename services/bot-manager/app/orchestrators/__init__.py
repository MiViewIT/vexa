"""Orchestrator module loader.

This module dynamically loads the appropriate orchestrator implementation
based on the ORCHESTRATOR environment variable.

Supported orchestrators:
- docker (default): Uses Docker socket to spawn bot containers
- nomad: Uses HashiCorp Nomad to dispatch parameterized jobs
- process: Spawns bots as local Node.js processes (for Lite deployments)
- temporal: Uses Temporal workflows as durable control plane (Nomad-backed activities)
"""
import os
import importlib
import logging

logger = logging.getLogger("bot_manager.orchestrator_loader")

_orchestrator = os.getenv("ORCHESTRATOR", "docker").lower()

# Select the appropriate orchestrator module based on ORCHESTRATOR env var
if _orchestrator == "nomad":
    module_name = "app.orchestrators.nomad"
elif _orchestrator == "process":
    module_name = "app.orchestrators.process"
elif _orchestrator == "temporal":
    module_name = "app.orchestrators.temporal"
else:
    # Default to Docker orchestrator
    module_name = "app.orchestrators.docker"

logger.info(f"Using '{_orchestrator}' orchestrator module: {module_name}")

# Dynamically import the concrete module
mod = importlib.import_module(module_name)

# Re-export a stable interface expected by the rest of the codebase
get_socket_session = getattr(mod, "get_socket_session", lambda *args, **kwargs: None)
close_docker_client = getattr(mod, "close_docker_client", getattr(mod, "close_client", lambda: None))
start_bot_container = mod.start_bot_container  # type: ignore
stop_bot_container = getattr(mod, "stop_bot_container", lambda *args, **kwargs: None)
_record_session_start = getattr(mod, "_record_session_start", lambda *args, **kwargs: None)
get_running_bots_status = getattr(mod, "get_running_bots_status", lambda *args, **kwargs: {})
verify_container_running = getattr(mod, "verify_container_running", lambda *args, **kwargs: False)
start_meeting_workflow = getattr(mod, "start_meeting_workflow", None)
signal_stop_meeting = getattr(mod, "signal_stop_meeting", None)
query_meeting_workflow = getattr(mod, "query_meeting_workflow", None)
signal_reconfigure_meeting = getattr(mod, "signal_reconfigure_meeting", None)
signal_bot_status_update = getattr(mod, "signal_bot_status_update", None)
start_deferred_transcription_workflow = getattr(mod, "start_deferred_transcription_workflow", None)
check_orchestrator_health = getattr(mod, "check_orchestrator_health", None)
