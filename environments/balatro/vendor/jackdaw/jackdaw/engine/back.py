"""Runtime Back class for Balatro deck backs.

Ports the lifecycle methods from ``back.lua``:

* ``apply_to_run`` (back.lua:174) — starting conditions applied once at run
  start, after deck building.
* ``trigger_effect`` (back.lua:108) — in-game effects during scoring or round
  transitions.

Source references
-----------------
- back.lua:108  — ``Back:trigger_effect``
- back.lua:174  — ``Back:apply_to_run``
"""

from __future__ import annotations

import math
from typing import Any

from jackdaw.engine.data.prototypes import BACKS


class Back:
    """Runtime wrapper around a :class:`~jackdaw.engine.data.prototypes.BackProto`.

    Parameters
    ----------
    key:
        Center key for the deck back (e.g. ``'b_plasma'``, ``'b_red'``).
    """

    def __init__(self, key: str) -> None:
        self.key: str = key
        self.proto = BACKS[key]
        self.name: str = self.proto.name
        # config may be stored as a list (empty config) or dict
        cfg = self.proto.config
        self.config: dict[str, Any] = cfg if isinstance(cfg, dict) else {}

    # ------------------------------------------------------------------
    # apply_to_run — back.lua:174-278
    # ------------------------------------------------------------------

    def apply_to_run(self, game_state: dict[str, Any]) -> dict[str, Any]:
        """Apply deck starting conditions and return a mutations dict.

        Called once at run start, after
        :func:`~jackdaw.engine.deck_builder.build_deck`.  The returned dict
        describes every game-state field that should be modified.

        Mirrors ``Back:apply_to_run`` (``back.lua:174-278``).

        Parameters
        ----------
        game_state:
            Mutable game-state dict.  This method does **not** mutate it
            directly — callers apply the returned mutations.

        Returns
        -------
        dict
            Mutations to apply.  Keys:

            * ``hands_delta`` — additive hand count change
            * ``discards_delta`` — additive discard count change
            * ``hand_size_delta`` — additive hand-size change
            * ``joker_slots_delta`` — additive joker-slot change
            * ``consumable_slots_delta`` — additive consumable-slot change
            * ``dollars_delta`` — starting money bonus
            * ``starting_vouchers`` — list of voucher keys to apply at run start
            * ``starting_consumables`` — list of consumable keys to add
            * ``ante_scaling`` — ante scaling factor (Plasma Deck)
            * ``money_per_hand`` — dollars earned per hand played (Green Deck)
            * ``money_per_discard`` — dollars earned per discard (Green Deck)
            * ``no_interest`` — suppress end-of-round interest (Green Deck)
            * ``spectral_rate`` — base spectral draw rate (Ghost Deck)
            * ``reroll_discount`` — flat discount on reroll cost
        """
        config = self.config
        mutations: dict[str, Any] = {}

        # -- Hands / discards / hand size / slots --
        if "hands" in config:
            mutations["hands_delta"] = config["hands"]
        if "discards" in config:
            mutations["discards_delta"] = config["discards"]
        if "hand_size" in config:
            mutations["hand_size_delta"] = config["hand_size"]
        if "joker_slot" in config:
            mutations["joker_slots_delta"] = config["joker_slot"]
        if "consumable_slot" in config:
            mutations["consumable_slots_delta"] = config["consumable_slot"]

        # -- Starting money --
        if "dollars" in config:
            mutations["dollars_delta"] = config["dollars"]

        # -- Starting vouchers (single key or list) --
        if "voucher" in config:
            v = config["voucher"]
            mutations["starting_vouchers"] = [v] if isinstance(v, str) else list(v)
        if "vouchers" in config:
            mutations["starting_vouchers"] = list(config["vouchers"])

        # -- Starting consumables --
        if "consumables" in config:
            mutations["starting_consumables"] = list(config["consumables"])

        # -- Ante scaling (Plasma Deck: 2× ante difficulty) --
        if "ante_scaling" in config:
            mutations["ante_scaling"] = config["ante_scaling"]

        # -- Green Deck: per-hand/discard money bonus, no end-of-round interest --
        if "extra_hand_bonus" in config:
            mutations["money_per_hand"] = config["extra_hand_bonus"]
        if "extra_discard_bonus" in config:
            mutations["money_per_discard"] = config["extra_discard_bonus"]
        if config.get("no_interest"):
            mutations["no_interest"] = True

        # -- Ghost Deck: raised spectral appearance rate --
        if "spectral_rate" in config:
            mutations["spectral_rate"] = config["spectral_rate"]

        # -- Reroll discount (reserved; not in current centers.json) --
        if "reroll_discount" in config:
            mutations["reroll_discount"] = config["reroll_discount"]

        return mutations

    # ------------------------------------------------------------------
    # trigger_effect — back.lua:108-172
    # ------------------------------------------------------------------

    def trigger_effect(
        self,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Return an effect payload for in-game triggers, or ``None``.

        Mirrors ``Back:trigger_effect`` (``back.lua:108-172``).

        Parameters
        ----------
        context:
            One of:

            ``'final_scoring_step'``
                Called in Phase 10 of the scoring pipeline, after all joker
                effects and before card destruction.

                Kwargs: ``chips`` (float), ``mult`` (float).

                Return shape: ``{'chips': int, 'mult': int}`` — both set to
                ``floor((chips + mult) / 2)`` for Plasma Deck.

            ``'eval'``
                Called after blind evaluation (boss-blind defeat check).

                Kwargs: ``boss_defeated`` (bool).

                Return shape: ``{'create_tag': 'tag_double'}`` for Anaglyph
                Deck when *boss_defeated* is True.

        Returns
        -------
        dict | None
            Effect payload, or ``None`` if this back has no effect in the
            given context.
        """
        # -- Plasma Deck: average chips and mult --
        if context == "final_scoring_step" and self.key == "b_plasma":
            chips: float = kwargs["chips"]
            mult: float = kwargs["mult"]
            total = chips + mult
            averaged = math.floor(total / 2)
            return {"chips": averaged, "mult": averaged}

        # -- Anaglyph Deck: create Double Tag on boss defeat --
        if context == "eval" and self.key == "b_anaglyph":
            if kwargs.get("boss_defeated"):
                return {"create_tag": "tag_double"}

        return None
