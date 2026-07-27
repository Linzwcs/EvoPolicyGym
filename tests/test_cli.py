from __future__ import annotations

import argparse
import unittest

from evopolicygym._protocol.session import SESSION_MAX_EPISODE_INDICES
from evopolicygym.cli import _parse_episode_selector


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
