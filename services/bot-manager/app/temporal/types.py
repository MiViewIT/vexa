from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class MeetingWorkflowInput:
    meeting_id: int
    user_id: int
    meeting_url: Optional[str]
    platform: str
    bot_name: Optional[str]
    user_token: str
    native_meeting_id: str
    language: Optional[str]
    task: Optional[str]
    transcription_tier: str = "realtime"
    transcribe_enabled: Optional[bool] = None
    recording_enabled: Optional[bool] = None
    zoom_obf_token: Optional[str] = None
    voice_agent_enabled: Optional[bool] = None
    default_avatar_url: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BotStatusSignal:
    status: str
    reason: Optional[str] = None
    exit_code: Optional[int] = None
    container_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconfigureSignal:
    language: Optional[str] = None
    task: Optional[str] = None
    transcription_tier: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeferredTranscriptionInput:
    recording_id: int
    job_id: int
    meeting_id: Optional[int]
    user_id: int
    language: Optional[str]
    task: str = "transcribe"

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)
