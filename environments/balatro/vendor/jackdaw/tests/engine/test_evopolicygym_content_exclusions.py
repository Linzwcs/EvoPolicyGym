"""EvoPolicyGym regressions for pre-selection Tag and Voucher exclusions."""

from __future__ import annotations

import unittest

from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import assign_ante_blinds

_EXCLUDED_TAGS = {"tag_rare", "tag_uncommon", "tag_voucher"}
_EXCLUDED_VOUCHERS = {
    "v_omen_globe",
    "v_telescope",
    "v_observatory",
    "v_directors_cut",
    "v_retcon",
}
_EXCLUDED_CONTENT = _EXCLUDED_TAGS | _EXCLUDED_VOUCHERS


class ContentExclusionTests(unittest.TestCase):
    def test_assign_ante_blinds_forwards_banned_keys_to_both_pools(self) -> None:
        found_unfiltered_tag = False
        found_unfiltered_voucher = False

        for index in range(256):
            seed = f"EPG_CONTENT_EXCLUSION_{index}"
            unfiltered = assign_ante_blinds(
                8,
                PseudoRandom(seed),
                {"round_resets": {}},
            )
            found_unfiltered_tag = found_unfiltered_tag or any(
                key in _EXCLUDED_TAGS
                for key in unfiltered["blind_tags"].values()
            )
            found_unfiltered_voucher = (
                found_unfiltered_voucher
                or unfiltered["voucher"] in _EXCLUDED_VOUCHERS
            )

            filtered = assign_ante_blinds(
                8,
                PseudoRandom(seed),
                {
                    "banned_keys": {key: True for key in _EXCLUDED_CONTENT},
                    "round_resets": {},
                },
            )
            self.assertNotIn(filtered["voucher"], _EXCLUDED_VOUCHERS)
            self.assertTrue(
                _EXCLUDED_TAGS.isdisjoint(filtered["blind_tags"].values())
            )

        self.assertTrue(found_unfiltered_tag)
        self.assertTrue(found_unfiltered_voucher)


if __name__ == "__main__":
    unittest.main()
