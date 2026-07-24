from __future__ import annotations

import unittest

from evopolicygym._protocol.policy import (
    POLICY_FRAMES,
    POLICY_MAX_FRAME_BYTES,
    decode_policy_value,
    encode_policy_value,
)
from evopolicygym._protocol.session import (
    SESSION_FRAMES,
    SESSION_MAX_FRAME_BYTES,
)
from evopolicygym.policy import PolicyValue, TensorValue


class PolicyProtocolTests(unittest.TestCase):
    def test_policy_values_round_trip_without_changing_carriers(self) -> None:
        value: PolicyValue = {
            "none": None,
            "items": [True, 7, -3, 1.25, "text", b"bytes"],
            "tuple": (1, 2),
            "tensor": TensorValue(
                dtype="int16",
                shape=(2,),
                data=b"\x01\x00\xff\xff",
            ),
        }

        self.assertEqual(
            decode_policy_value(encode_policy_value(value)),
            value,
        )

    def test_policy_frame_classifies_partial_and_complete_malformed_input(self) -> None:
        frame = POLICY_FRAMES.encode({"type": "ready"})
        self.assertEqual(POLICY_FRAMES.decode(frame), {"type": "ready"})
        with self.assertRaises(EOFError):
            POLICY_FRAMES.decode(frame[:-1])
        malformed = (1).to_bytes(4, "big") + b"{"
        with self.assertRaises(ValueError):
            POLICY_FRAMES.decode(malformed)
        oversized = (POLICY_MAX_FRAME_BYTES + 1).to_bytes(4, "big")
        with self.assertRaises(ValueError):
            POLICY_FRAMES.decode(oversized)


class AgentSessionProtocolTests(unittest.TestCase):
    def test_session_frame_round_trip_and_boundaries(self) -> None:
        message = {
            "protocol": "agent-session/v1",
            "method": "submit",
            "episodes": 3,
        }
        frame = SESSION_FRAMES.encode(message)
        self.assertEqual(SESSION_FRAMES.decode(frame), message)
        with self.assertRaises(EOFError):
            SESSION_FRAMES.decode(frame[:2])
        with self.assertRaises(EOFError):
            SESSION_FRAMES.decode(frame[:-1])
        with self.assertRaises(ValueError):
            SESSION_FRAMES.decode(frame + b"x")
        with self.assertRaises(ValueError):
            SESSION_FRAMES.decode_payload(b"")
        with self.assertRaises(ValueError):
            SESSION_FRAMES.decode_payload(b'{"value":NaN}')
        with self.assertRaises(ValueError):
            SESSION_FRAMES.decode_payload(
                b"x" * (SESSION_MAX_FRAME_BYTES + 1)
            )
        oversized = (SESSION_MAX_FRAME_BYTES + 1).to_bytes(4, "big")
        with self.assertRaises(ValueError):
            SESSION_FRAMES.decode(oversized)


if __name__ == "__main__":
    unittest.main()
