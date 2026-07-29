from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from evopolicygym._protocol.session import SESSION_MAX_EPISODE_INDICES
from evopolicygym.cli import main as operator_main
from evopolicygym.run._session_cli import _parse_episode_selector


class OperatorCommandTests(unittest.TestCase):
    def test_operator_command_does_not_expose_agent_session_methods(self) -> None:
        standard_output = io.StringIO()
        with redirect_stdout(standard_output):
            self.assertEqual(operator_main([]), 0)
        self.assertIn("public Python SDK", standard_output.getvalue())

        standard_error = io.StringIO()
        with redirect_stderr(standard_error):
            with self.assertRaises(SystemExit) as raised:
                operator_main(["submit"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: submit", standard_error.getvalue())


class EpisodeSelectorTests(unittest.TestCase):
    def test_ranges_and_singletons_expand_to_a_canonical_union(self) -> None:
        self.assertEqual(
            _parse_episode_selector("0:2,4:8"),
            (0, 1, 4, 5, 6, 7),
        )
        self.assertEqual(
            _parse_episode_selector("0,2,5"),
            (0, 2, 5),
        )
        self.assertEqual(
            _parse_episode_selector("0:2,2:4"),
            (0, 1, 2, 3),
        )

    def test_malformed_duplicate_and_overlapping_selectors_are_rejected(
        self,
    ) -> None:
        for selector in (
            "",
            "0,",
            ",0",
            "0 1",
            "-1",
            "0:0",
            "2:1",
            "0::1",
            "0,0",
            "0:3,2:4",
            "2,1",
            str(2**64),
            f"0:{2**64 + 1}",
        ):
            with self.subTest(selector=selector):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_episode_selector(selector)

    def test_selector_has_a_protocol_expansion_limit(self) -> None:
        accepted = _parse_episode_selector(
            f"0:{SESSION_MAX_EPISODE_INDICES}"
        )
        self.assertEqual(len(accepted), SESSION_MAX_EPISODE_INDICES)
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_episode_selector(
                f"0:{SESSION_MAX_EPISODE_INDICES + 1}"
            )


if __name__ == "__main__":
    unittest.main()
