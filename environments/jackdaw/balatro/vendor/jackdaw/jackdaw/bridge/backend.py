"""Swappable backend interface for the JSON-RPC server.

Two implementations:

- **SimBackend** — runs the jackdaw engine in-process. Fast, headless,
  deterministic, zero runtime deps.
- **LiveBackend** — proxies all requests to a real balatrobot instance
  over HTTP. Requires a running Balatro + balatrobot mod.
"""

from __future__ import annotations

from typing import Any, Protocol

from jackdaw.engine.actions import GamePhase

# ---------------------------------------------------------------------------
# Balatrobot enum → engine value maps
# ---------------------------------------------------------------------------

DECK_FROM_BOT: dict[str, str] = {
    "RED": "b_red",
    "BLUE": "b_blue",
    "YELLOW": "b_yellow",
    "GREEN": "b_green",
    "BLACK": "b_black",
    "MAGIC": "b_magic",
    "NEBULA": "b_nebula",
    "GHOST": "b_ghost",
    "ABANDONED": "b_abandoned",
    "CHECKERED": "b_checkered",
    "ZODIAC": "b_zodiac",
    "PAINTED": "b_painted",
    "ANAGLYPH": "b_anaglyph",
    "PLASMA": "b_plasma",
    "ERRATIC": "b_erratic",
}

STAKE_FROM_BOT: dict[str, int] = {
    "WHITE": 1,
    "RED": 2,
    "GREEN": 3,
    "BLACK": 4,
    "BLUE": 5,
    "PURPLE": 6,
    "ORANGE": 7,
    "GOLD": 8,
}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RPCError(Exception):
    """JSON-RPC error with code, message, and optional data."""

    def __init__(
        self,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)


# Error codes
BAD_REQUEST = -32001
INVALID_STATE = -32002
NOT_ALLOWED = -32003


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Structural interface for a JSON-RPC backend."""

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle a JSON-RPC method call. Returns the result dict."""
        ...


# ---------------------------------------------------------------------------
# SimBackend
# ---------------------------------------------------------------------------

# Methods that map to engine game actions (handled via rpc_to_action + step).
_ACTION_METHODS = frozenset(
    {
        "play",
        "discard",
        "select",
        "skip",
        "buy",
        "sell",
        "use",
        "reroll",
        "next_round",
        "cash_out",
        "pack",
        "rearrange",
    }
)


class SimBackend:
    """Backend that runs the jackdaw engine in-process."""

    def __init__(self) -> None:
        self._gs: dict[str, Any] | None = None

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            params = {}

        if method == "health":
            return {"status": "ok"}

        if method == "start":
            return self._handle_start(params)

        if method == "menu":
            self._gs = None
            return {"state": "MENU"}

        if method == "gamestate":
            return self._require_gamestate()

        if method == "add":
            return self._handle_add(params)

        if method == "set":
            return self._handle_set(params)

        if method in _ACTION_METHODS:
            return self._handle_action(method, params)

        raise RPCError(BAD_REQUEST, f"Unknown method: {method!r}")

    # -- internal -----------------------------------------------------------

    def _handle_start(self, params: dict[str, Any]) -> dict[str, Any]:
        from jackdaw.engine.run_init import initialize_run

        deck_str = params.get("deck", "RED")
        stake_str = params.get("stake", "WHITE")
        seed = params.get("seed", "DEFAULT")

        back_key = DECK_FROM_BOT.get(deck_str, "b_red")
        stake = STAKE_FROM_BOT.get(stake_str, 1)

        self._gs = initialize_run(back_key, stake, seed)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"

        return self._serialize()

    def _handle_action(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        from jackdaw.bridge.deserializer import rpc_to_action

        try:
            action = rpc_to_action(method, params)
        except ValueError as exc:
            raise RPCError(BAD_REQUEST, str(exc)) from exc

        if action is None:
            return self._serialize()

        from jackdaw.engine.game import IllegalActionError, step

        try:
            step(self._gs, action)
        except IllegalActionError as exc:
            raise RPCError(NOT_ALLOWED, str(exc)) from exc

        return self._serialize()

    def _handle_add(self, params: dict[str, Any]) -> dict[str, Any]:
        """Insert a card into the game (debug method matching balatrobot API)."""
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        key = params.get("key")
        if not key:
            raise RPCError(BAD_REQUEST, "'add' requires a 'key' parameter")

        # Validate phase (matches balatrobot API constraints)
        phase = self._gs.get("phase")
        if key.startswith("v_") or key.startswith("p_"):
            if phase != GamePhase.SHOP:
                raise RPCError(
                    INVALID_STATE,
                    f"Method 'add' requires SHOP state for vouchers/packs (got {phase})",
                )
        elif phase not in (GamePhase.SELECTING_HAND, GamePhase.SHOP, GamePhase.ROUND_EVAL):
            raise RPCError(
                INVALID_STATE,
                "Method 'add' requires one of these states: SELECTING_HAND, SHOP, ROUND_EVAL",
            )

        from jackdaw.engine.card_factory import (
            RANK_LETTER,
            SUIT_LETTER,
            create_consumable,
            create_joker,
            create_playing_card,
        )
        from jackdaw.engine.card_utils import poll_edition

        seal = params.get("seal")  # Gold, Red, Blue, Purple
        edition_key = params.get("edition")  # foil, holo, polychrome, negative
        enhancement = params.get("enhancement", "c_base")  # m_bonus, m_mult, etc.
        eternal = params.get("eternal", False)
        perishable = params.get("perishable", False)
        rental = params.get("rental", False)

        edition = {edition_key: True} if edition_key else None

        if key.startswith("j_"):
            card = create_joker(
                key,
                edition=edition,
                eternal=eternal,
                perishable=perishable,
                rental=rental,
                hands_played=self._gs.get("hands_played", 0),
            )
            # Poll for edition to match balatrobot's create_card() behaviour.
            # Balatrobot's add command goes through the full Lua create_card()
            # pipeline which calls poll_edition(), so we must do the same to
            # keep the RNG stream in sync.
            rng = self._gs.get("rng")
            if rng is not None:
                ante = self._gs.get("round_resets", {}).get("ante", 1)
                polled = poll_edition(
                    "edi" + str(ante),
                    rng,
                    rate=self._gs.get("edition_rate", 1.0),
                )
                if edition is None:
                    card.set_edition(polled)
            card.set_cost()
            self._gs["jokers"].append(card)
            # Apply passive effects (hand_size, discards, joker_slots, etc.)
            old_hand_size = self._gs.get("hand_size", 8)
            card.add_to_deck(self._gs)
            # Also update current round counters for mid-round adds.
            # Only discards are adjusted immediately — extra discards are
            # usable in the current round.  Hand count changes (h_plays)
            # take effect on future rounds via round_resets (already set
            # by add_to_deck), matching balatrobot behaviour.
            cr = self._gs.get("current_round")
            if cr is not None:
                d_size = card.ability.get("d_size", 0)
                if d_size > 0:
                    cr["discards_left"] = cr.get("discards_left", 0) + d_size
            # If hand_size increased mid-round, draw cards to fill the new
            # size — matches balatrobot which immediately fills on add.
            if self._gs.get("hand_size", 8) > old_hand_size and phase == GamePhase.SELECTING_HAND:
                from jackdaw.engine.game import _draw_hand

                _draw_hand(self._gs)
        elif key.startswith("c_"):
            card = create_consumable(key)
            self._gs["consumables"].append(card)
        elif len(key) == 3 and key[1] == "_":
            # Playing card key like "H_A", "S_2"
            suit_letter, rank_letter = key[0], key[2]
            if suit_letter not in SUIT_LETTER or rank_letter not in RANK_LETTER:
                raise RPCError(BAD_REQUEST, f"Invalid playing card key: {key!r}")
            suit = SUIT_LETTER[suit_letter]
            rank = RANK_LETTER[rank_letter]
            card = create_playing_card(
                suit=suit,
                rank=rank,
                enhancement=enhancement,
                edition=edition,
                seal=seal,
            )
            # Add to hand if in SELECTING_HAND, otherwise to deck
            phase = self._gs.get("phase")
            if phase == GamePhase.SELECTING_HAND:
                self._gs["hand"].append(card)
            else:
                self._gs["deck"].append(card)
        else:
            raise RPCError(BAD_REQUEST, f"Unrecognised card key prefix: {key!r}")

        return self._serialize()

    def _handle_set(self, params: dict[str, Any]) -> dict[str, Any]:
        """Modify game state values (debug method matching balatrobot API)."""
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        if "money" in params:
            self._gs["dollars"] = params["money"]

        if "hands" in params:
            self._gs["current_round"]["hands_left"] = params["hands"]

        if "discards" in params:
            self._gs["current_round"]["discards_left"] = params["discards"]

        if "ante" in params:
            self._gs["round_resets"]["ante"] = params["ante"]

        if "round" in params:
            self._gs["round"] = params["round"]

        if "chips" in params:
            self._gs["chips"] = params["chips"]

        if "shop" in params and params["shop"]:
            from jackdaw.engine.shop import populate_shop

            populate_shop(self._gs)

        return self._serialize()

    def _require_gamestate(self) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")
        return self._serialize()

    def _serialize(self) -> dict[str, Any]:
        from jackdaw.bridge.serializer import game_state_to_bot_response

        return game_state_to_bot_response(self._gs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LiveBackend
# ---------------------------------------------------------------------------


class LiveBackend:
    """Backend that proxies requests to a real balatrobot instance."""

    # Internal → balatrobot format maps for the "add" method
    _ENHANCEMENT_TO_BOT: dict[str, str] = {
        "m_bonus": "BONUS",
        "m_mult": "MULT",
        "m_wild": "WILD",
        "m_glass": "GLASS",
        "m_steel": "STEEL",
        "m_stone": "STONE",
        "m_gold": "GOLD",
        "m_lucky": "LUCKY",
    }
    _EDITION_TO_BOT: dict[str, str] = {
        "foil": "FOIL",
        "holo": "HOLO",
        "polychrome": "POLYCHROME",
        "negative": "NEGATIVE",
    }
    _SEAL_TO_BOT: dict[str, str] = {
        "Gold": "GOLD",
        "Red": "RED",
        "Blue": "BLUE",
        "Purple": "PURPLE",
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 12346) -> None:
        self._url = f"http://{host}:{port}"

    def _convert_add_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert internal format to balatrobot format for 'add' params."""
        params = dict(params)  # shallow copy
        if "enhancement" in params:
            params["enhancement"] = self._ENHANCEMENT_TO_BOT.get(
                params["enhancement"], params["enhancement"]
            )
        if "edition" in params:
            params["edition"] = self._EDITION_TO_BOT.get(params["edition"], params["edition"])
        if "seal" in params:
            params["seal"] = self._SEAL_TO_BOT.get(params["seal"], params["seal"])
        return params

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        import httpx

        rpc_params = params or {}
        if method == "add" and rpc_params:
            rpc_params = self._convert_add_params(rpc_params)

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": rpc_params,
            "id": 1,
        }
        resp = httpx.post(self._url, json=payload, timeout=10.0)
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RPCError(
                code=err.get("code", -32000),
                message=err.get("message", "Unknown error"),
                data=err.get("data", {}),
            )

        return data.get("result", {})
