import unittest

from app.temporal.types import DeferredTranscriptionInput, MeetingWorkflowInput


class TemporalTypesTests(unittest.TestCase):
    def test_meeting_workflow_input_payload(self):
        payload = MeetingWorkflowInput(
            meeting_id=101,
            user_id=202,
            meeting_url="https://meet.example/test",
            platform="google_meet",
            bot_name="Vexa",
            user_token="token",
            native_meeting_id="abc-defg-hij",
            language="en",
            task="transcribe",
            transcription_tier="realtime",
            transcribe_enabled=True,
            recording_enabled=False,
        ).to_payload()

        self.assertEqual(payload["meeting_id"], 101)
        self.assertEqual(payload["user_id"], 202)
        self.assertEqual(payload["transcription_tier"], "realtime")
        self.assertTrue(payload["transcribe_enabled"])
        self.assertFalse(payload["recording_enabled"])

    def test_deferred_transcription_payload(self):
        payload = DeferredTranscriptionInput(
            recording_id=12,
            job_id=34,
            meeting_id=56,
            user_id=78,
            language="en",
            task="transcribe",
        ).to_payload()

        self.assertEqual(payload["recording_id"], 12)
        self.assertEqual(payload["job_id"], 34)
        self.assertEqual(payload["meeting_id"], 56)
        self.assertEqual(payload["user_id"], 78)


if __name__ == "__main__":
    unittest.main()
