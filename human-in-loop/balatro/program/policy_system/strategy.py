"""Phase planning and shared action selection."""

from __future__ import annotations

from typing import Any, cast

from .actions import ActionGateway
from .boss import filter_candidates, first_hand_hidden, only_one_hand
from .build import (
    acquisition_score,
    choose_joker_upgrade,
    effect_context,
    value_joker,
)
from .consumables import targeted_consumable
from .economy import (
    EconomyContext,
    choose_celestial,
    choose_planet_purchase,
    choose_reroll,
    choose_voucher,
)
from .effects import next_main_xmult_swap
from .hands import candidates, cycle_play_choice, discard_choice
from .state import EpisodePlan, StateView


class BalatroPolicy:
    def __init__(self) -> None:
        self.plan = EpisodePlan()

    def act(self, observation: Any) -> Any:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        state = StateView(cast(dict[str, Any], observation))
        gate = ActionGateway(state.legal)
        if state.phase == "blind_select":
            self.plan.used_discard_this_round = False
            self.plan.prepare_round(int(state.progress.get("rounds_cleared", 0)))
            return gate.simple("select_blind")
        if state.phase == "round_eval":
            return gate.simple("cash_out")
        if state.phase == "selecting_hand":
            return self._hand(state, gate)
        if state.phase == "shop":
            return self._shop(state, gate)
        if state.phase == "pack_opening":
            return self._pack(state, gate)
        raise RuntimeError(f"unexpected phase {state.phase!r}")

    def _hand(
        self,
        state: StateView,
        gate: ActionGateway,
    ) -> dict[str, Any]:
        self.plan.prepare_round(int(state.progress.get("rounds_cleared", 0)))
        consumable = targeted_consumable(
            state.consumables,
            state.legal.get("use_consumable", {}),
            state.hand,
            self.plan.primary_hand,
        )
        if consumable is not None:
            target, indices = consumable
            action = gate.target_cards("use_consumable", target, indices)
            if action is not None:
                return action
        arrangement = self._arrange_copy_joker(state, gate)
        if arrangement is not None:
            return arrangement
        arrangement = self._arrange_main_xmult(state, gate)
        if arrangement is not None:
            return arrangement
        ranked = candidates(
            state.hand,
            state.poker_hands,
            state.jokers,
            money=state.money(),
            deck_remaining=int(state.deck.get("draw_pile", 0)),
            discards_left=int(state.resources.get("discards_left", 0)),
            hands_left=int(state.resources.get("hands_left", 0)),
            round_hand_counts=self.plan.played_hand_counts,
        )
        ranked = filter_candidates(
            ranked,
            blind=state.blind,
            played_hand_types=self.plan.played_hand_types or set(),
        )
        score, name, indices = ranked[0]
        self.plan.primary_hand = name
        hands = int(state.resources.get("hands_left", 1))
        discards = int(state.resources.get("discards_left", 0))
        draw_threshold = 0.82 + 0.10 * min(discards, 4)
        if (
            int(state.progress.get("ante", 1)) <= 2
            and not state.jokers
            and self.plan.consecutive_discards >= 1
        ):
            draw_threshold = min(draw_threshold, 0.90)
        if (
            discards
            and (hands > 1 or only_one_hand(state.blind))
            and not first_hand_hidden(state.blind, state.hand)
            and score < state.remaining()
            and (
                name == "High Card"
                or score
                < state.remaining() / hands * draw_threshold
            )
        ):
            choice = discard_choice(state.hand, indices)
            action = gate.cards("discard", choice)
            if action:
                self.plan.used_discard_this_round = True
                self.plan.consecutive_discards += 1
                return action
        play_indices = indices
        if (
            hands > 1
            and discards == 0
            and int(state.progress.get("ante", 1)) >= 2
            and score < state.remaining()
        ):
            play_indices = cycle_play_choice(
                state.hand,
                ranked,
                score=score,
                hand_name=name,
                selected_indices=indices,
            )
        action = gate.cards("play_hand", play_indices)
        if action:
            self.plan.consecutive_discards = 0
            if self.plan.played_hand_types is None:
                self.plan.played_hand_types = set()
            self.plan.played_hand_types.add(name)
            if self.plan.played_hand_counts is None:
                self.plan.played_hand_counts = {}
            self.plan.played_hand_counts[name] = (
                self.plan.played_hand_counts.get(name, 0) + 1
            )
            return action
        raise RuntimeError("no admitted hand action")

    def _arrange_copy_joker(
        self,
        state: StateView,
        gate: ActionGateway,
    ) -> dict[str, Any] | None:
        copy_positions = [
            position
            for position, joker in enumerate(state.jokers)
            if "copy the joker to the right"
            in str(
                (joker.get("rule") or {}).get("summary", "")
            ).lower()
        ]
        if not copy_positions:
            return None

        copy_position = copy_positions[0]
        context = effect_context(
            primary_hand=self.plan.primary_hand,
            money=state.money(),
            jokers=state.jokers,
            deck_remaining=int(state.deck.get("draw_pile", 0)),
            ante=int(state.progress.get("ante", 1)),
        )
        targets = [
            (
                value_joker(joker, context, owned=True).value,
                position,
                joker,
            )
            for position, joker in enumerate(state.jokers)
            if position not in copy_positions
        ]
        if not targets:
            return None
        if state.hand:
            copy_joker = state.jokers[copy_position]
            non_copy = [
                joker
                for position, joker in enumerate(state.jokers)
                if position != copy_position
            ]
            scoring_context: dict[str, Any] = {
                "money": state.money(),
                "deck_remaining": int(
                    state.deck.get("draw_pile", 0)
                ),
                "discards_left": int(
                    state.resources.get("discards_left", 0)
                ),
                "hands_left": int(
                    state.resources.get("hands_left", 0)
                ),
                "round_hand_counts": self.plan.played_hand_counts,
            }

            def copied_score(target: dict[str, Any]) -> float:
                position = next(
                    index
                    for index, joker in enumerate(non_copy)
                    if joker is target
                )
                arranged = list(non_copy)
                arranged.insert(position, copy_joker)
                ranked = candidates(
                    state.hand,
                    state.poker_hands,
                    arranged,
                    **scoring_context,
                )
                return ranked[0][0]

            _, target_position, _ = max(
                targets,
                key=lambda item: copied_score(item[2]),
            )
        else:
            _, target_position, _ = max(targets)
        if copy_position + 1 == target_position:
            return None

        copy_index = int(
            state.jokers[copy_position].get("index", copy_position)
        )
        kind = (
            "swap_joker_left"
            if copy_position > target_position
            else "swap_joker_right"
        )
        return gate.entity(kind, copy_index)

    def _arrange_main_xmult(
        self,
        state: StateView,
        gate: ActionGateway,
    ) -> dict[str, Any] | None:
        position = next_main_xmult_swap(state.jokers)
        if position is None:
            return None
        index = int(state.jokers[position].get("index", position))
        return gate.entity("swap_joker_right", index)

    def _shop(
        self,
        state: StateView,
        gate: ActionGateway,
    ) -> dict[str, Any]:
        money = state.money()
        slots = int(state.resources.get("joker_slots", 5))
        shop_cards = list(state.shop.get("cards", []))
        round_id = int(state.progress.get("rounds_cleared", 0))
        if self.plan.shop_round != round_id:
            self.plan.shop_round = round_id
            self.plan.rerolls_this_shop = 0
            self.plan.celestials_this_shop = 0
            self.plan.planets_this_shop = 0

        if self.plan.pending_purchase_key:
            pending = next(
                (
                    card
                    for card in shop_cards
                    if card.get("key") == self.plan.pending_purchase_key
                ),
                None,
            )
            if pending is not None:
                action = gate.entity("buy_card", int(pending.get("index", -1)))
                if action:
                    self.plan.pending_purchase_key = None
                    return action
            self.plan.pending_purchase_key = None

        sell_descriptor = state.legal.get("sell_joker", {})
        sellable = {
            int(index) for index in sell_descriptor.get("target_indices", [])
        }
        context = effect_context(
            primary_hand=self.plan.primary_hand,
            money=money,
            jokers=state.jokers,
            deck_remaining=int(state.deck.get("draw_pile", 0)),
            ante=int(state.progress.get("ante", 1)),
        )
        upgrade = choose_joker_upgrade(
            shop_cards=shop_cards,
            owned_cards=state.jokers,
            sellable_indices=sellable,
            slots=slots,
            money=money,
            context=context,
        )
        if upgrade is not None and upgrade.replaced is not None:
            replacement_index = int(upgrade.replaced.card.get("index", -1))
            action = gate.entity("sell_joker", replacement_index)
            if action:
                self.plan.pending_purchase_key = str(
                    upgrade.candidate.card.get("key", "")
                )
                return action
        if upgrade is not None:
            candidate_index = int(upgrade.candidate.card.get("index", -1))
            action = gate.entity("buy_card", candidate_index)
            if action:
                return action
        for booster in state.shop.get("boosters", []):
            name = str(booster.get("name", "")).lower()
            if (
                int(booster.get("cost", 99)) <= money
                and "buffoon" in name
            ):
                action = gate.entity(
                    "open_booster",
                    int(booster["index"]),
                )
                if action:
                    return action

        economy = EconomyContext(
            money=money,
            ante=int(state.progress.get("ante", 1)),
            reroll_cost=int(state.resources.get("reroll_cost", 5)),
            free_rerolls=int(state.resources.get("free_rerolls", 0)),
            rerolls_this_shop=self.plan.rerolls_this_shop,
            celestials_this_shop=self.plan.celestials_this_shop,
            joker_count=len(state.jokers),
            joker_slots=slots,
        )
        buy_descriptor = state.legal.get("buy_card", {})
        planet = choose_planet_purchase(
            shop_cards,
            legal_indices={
                int(index)
                for index in buy_descriptor.get("target_indices", [])
            },
            context=economy,
            primary_hand=self.plan.primary_hand,
            poker_hands=state.poker_hands,
            consumable_count=len(state.consumables),
            consumable_slots=int(
                state.resources.get("consumable_slots", 0)
            ),
            planets_this_shop=self.plan.planets_this_shop,
        )
        if planet is not None and planet.target_index is not None:
            action = gate.entity(planet.kind, planet.target_index)
            if action:
                self.plan.planets_this_shop += 1
                return action

        voucher_descriptor = state.legal.get("redeem_voucher", {})
        voucher = choose_voucher(
            list(state.shop.get("vouchers", [])),
            legal_indices={
                int(index)
                for index in voucher_descriptor.get("target_indices", [])
            },
            context=economy,
        )
        if voucher is not None and voucher.target_index is not None:
            action = gate.entity(voucher.kind, voucher.target_index)
            if action:
                return action

        booster_descriptor = state.legal.get("open_booster", {})
        celestial = choose_celestial(
            list(state.shop.get("boosters", [])),
            legal_indices={
                int(index)
                for index in booster_descriptor.get("target_indices", [])
            },
            context=economy,
        )
        if celestial is not None and celestial.target_index is not None:
            action = gate.entity(celestial.kind, celestial.target_index)
            if action:
                self.plan.celestials_this_shop += 1
                return action

        reroll = choose_reroll(
            reroll_legal="reroll_shop" in state.legal,
            context=economy,
        )
        if reroll is not None:
            action = gate.simple(reroll.kind)
            if action:
                self.plan.rerolls_this_shop += 1
                return action
        result = gate.simple("next_round")
        if result:
            return result
        raise RuntimeError("no admitted shop action")

    def _pack(
        self,
        state: StateView,
        gate: ActionGateway,
    ) -> dict[str, Any]:
        cards = list(state.pack.get("cards", []))
        hand_priorities = {
            str(hand.get("name", "")): (
                int(hand.get("played", 0)) * 2
                + max(0, int(hand.get("level", 1)) - 1) * 4
            )
            for hand in state.poker_hands.values()
        }

        def hand_alignment(card: dict[str, Any]) -> int:
            summary = str((card.get("rule") or {}).get("summary", "")).lower()
            return max(
                (
                    priority
                    for hand, priority in hand_priorities.items()
                    if hand.lower() in summary
                ),
                default=0,
            )

        cards.sort(
            key=lambda card: (
                hand_alignment(card),
                self.plan.primary_hand.lower()
                in str(
                    (card.get("rule") or {}).get("summary", "")
                ).lower(),
                acquisition_score(card),
                card.get("set") == "Joker",
            ),
            reverse=True,
        )
        desc = state.legal.get("pick_pack_card", {})
        advertised = {
            int(target) for target in desc.get("target_indices", [])
        }
        for card in cards:
            target = int(card.get("index", -1))
            if advertised and target not in advertised:
                continue
            action = gate.target_cards(
                "pick_pack_card",
                target,
                [],
            )
            if action:
                return action
        result = gate.simple("skip_pack")
        if result:
            return result
        raise RuntimeError("no admitted pack action")
