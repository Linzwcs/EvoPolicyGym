# EvoPolicyGym Jackdaw vendor record

This directory is a maintained source copy of
[TylerFlar/jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro),
licensed under the included MIT `LICENSE`.

## Source stack

The source was exported from this immutable upstream base:

- `c84dca9227b40eb5f7ff9fd7cd78945aa07854ce` — package-data fix from
  [PR #1](https://github.com/TylerFlar/jackdaw-balatro/pull/1).

The following upstream commits are applied in order:

1. `aaf24f93b4f22d3ee70a9099a211a7a6a93bef7e` — ten live-validated
   RNG, pool, targeting, and card-effect fixes from
   [PR #2](https://github.com/TylerFlar/jackdaw-balatro/pull/2).
2. `8e807df73797b500b1eccbdf26288f777619928c` — Certificate and Marble
   Joker playing-card front creation from
   [PR #6](https://github.com/TylerFlar/jackdaw-balatro/pull/6).
3. `8dd66169014b58b7a077760ff1090efe1d4a022c` — deferred skip-tag hooks
   and pack Joker-slot gating from PR #6.
4. `a785574bc6deea1c71cd53fec5b102bb82d52e8f` — Lua-compatible
   nan-collapse behavior for special seeds from
   [PR #5](https://github.com/TylerFlar/jackdaw-balatro/pull/5).

The resulting benchmark revision identifier is:

```text
c84dca9+aaf24f9+8e807df+8dd6616+a785574+epg1
```

## Local maintenance

EvoPolicyGym-specific regression tests live alongside the upstream tests and
use a `test_evopolicygym_*.py` filename. The vendored copy also contains
import-only Ruff cleanups in `engine/game.py` and `test_tag_wiring.py`; these
do not change runtime behavior. Preserve upstream commit identity in this
record whenever another patch is absorbed.

The `epg1` local patch threads `game_state.banned_keys` through Ante Tag and
Voucher pool construction. It uses Jackdaw's existing `UNAVAILABLE` sentinel
semantics, so excluded objects are filtered before RNG-backed selection. The
Balatro Benchmark uses this plumbing for its versioned active-content profile;
the vendored engine does not impose those exclusions by default.

This stack is not a claim of complete official Balatro equivalence. In
particular, upstream PR #2 reports two unresolved multi-Ante pool divergences,
and PR #6 deliberately leaves Rare/Uncommon Tag creation and Voucher Tag
replacement unwired pending live RNG validation.
