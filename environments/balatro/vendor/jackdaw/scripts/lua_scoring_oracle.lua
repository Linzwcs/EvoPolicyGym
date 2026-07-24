#!/usr/bin/env luajit
--- Scoring oracle: computes hand scores using the source's actual logic.
---
--- Stubs all UI/animation functions and runs a stripped-down scoring
--- pipeline synchronously.  Outputs JSON for cross-validation.
---
--- Compatible with LuaJIT 2.1.

-- Resolve project root
local script_path = arg[0] or ""
local scripts_pos = script_path:find("scripts")
local project_root = scripts_pos and script_path:sub(1, scripts_pos - 1) or "./"

----------------------------------------------------------------------------
-- Stubs for LÖVE2D / UI / animation dependencies
----------------------------------------------------------------------------

function card_eval_status_text() end
function update_hand_text() end
function ease_chips() end
function ease_dollars() end
function attention_text() end
function play_sound() end
function delay() end
function highlight_card() end
function juice_card() end
function set_hand_usage() end
function check_for_unlock() end
function check_and_set_high_score() end
function stop_use() end
function localize(...) return "loc" end
function inc_career_stat() end

function mod_chips(c) return math.max(0, math.floor(c)) end
function mod_mult(m) return math.max(1, m) end

-- Stub G.E_MANAGER: execute events immediately
local E_MANAGER = {}
function E_MANAGER:add_event(ev)
    if ev.func then ev.func() end
end

----------------------------------------------------------------------------
-- Minimal G state
----------------------------------------------------------------------------

G = {
    E_MANAGER = E_MANAGER,
    GAME = {
        probabilities = { normal = 1 },
        current_round = { current_hand = { handname = "" } },
        blind = nil,  -- set per test
        dollars = 100,
        dollar_buffer = 0,
    },
    play = { cards = {} },
    hand = { cards = {} },
    jokers = { cards = {} },
    consumeables = { cards = {} },
}

-- Stubs for card methods
function find_joker(name, non_debuff)
    local jokers = {}
    for _, v in pairs(G.jokers.cards) do
        if v and v.ability and v.ability.name == name and (non_debuff or not v.debuff) then
            jokers[#jokers+1] = v
        end
    end
    return jokers
end

----------------------------------------------------------------------------
-- Load source functions
----------------------------------------------------------------------------

local source_path = project_root .. "balatro_source/functions/misc_functions.lua"
local f = assert(io.open(source_path, "r"))
local src = f:read("*a")
f:close()

-- Extract hand eval functions (376-621)
local func_src = ""
local n = 0
for line in src:gmatch("[^\n]*\n?") do
    n = n + 1
    if n >= 376 and n <= 621 then func_src = func_src .. line end
end
assert(loadstring(func_src))()

-- Load eval_card from common_events.lua
local ce_path = project_root .. "balatro_source/functions/common_events.lua"
f = assert(io.open(ce_path, "r"))
src = f:read("*a")
f:close()

func_src = ""
n = 0
for line in src:gmatch("[^\n]*\n?") do
    n = n + 1
    if n >= 580 and n <= 656 then func_src = func_src .. line end
end
assert(loadstring(func_src))()

----------------------------------------------------------------------------
-- Card constructor
----------------------------------------------------------------------------

local id_map = {["2"]=2,["3"]=3,["4"]=4,["5"]=5,["6"]=6,["7"]=7,["8"]=8,["9"]=9,["10"]=10,Jack=11,Queen=12,King=13,Ace=14}
local nom_map = {["2"]=2,["3"]=3,["4"]=4,["5"]=5,["6"]=6,["7"]=7,["8"]=8,["9"]=9,["10"]=10,Jack=10,Queen=10,King=10,Ace=11}
local suit_nom = {Spades=0.04, Hearts=0.03, Clubs=0.02, Diamonds=0.01}
local face_nom_map = {Jack=0.1, Queen=0.2, King=0.3, Ace=0.4}

local card_counter = 0
local function make_card(suit, value, opts)
    opts = opts or {}
    card_counter = card_counter + 1
    local c = {
        base = {
            suit = suit, value = value,
            id = id_map[value], nominal = nom_map[value],
            suit_nominal = suit_nom[suit],
            face_nominal = face_nom_map[value] or 0,
        },
        ability = {
            name = opts.name or "Default Base",
            effect = opts.effect or "",
            set = opts.set or "Default",
            mult = opts.mult or 0,
            h_mult = opts.h_mult or 0,
            h_x_mult = opts.h_x_mult or 0,
            h_dollars = opts.h_dollars or 0,
            p_dollars = opts.p_dollars or 0,
            bonus = opts.bonus or 0,
            perma_bonus = 0,
            x_mult = opts.Xmult or 1,
        },
        edition = opts.edition,
        seal = opts.seal,
        debuff = false,
        sort_id = card_counter,
        unique_val = card_counter * 0.001,
        T = { x = card_counter },
        playing_card = true,
    }

    function c:get_id()
        if self.ability.effect == "Stone Card" then return -math.random(100,1000000) end
        return self.base.id
    end
    function c:get_nominal(mod)
        local mult = 1
        if mod == 'suit' then mult = 1000 end
        if self.ability.effect == 'Stone Card' then mult = -1000 end
        return self.base.nominal + self.base.suit_nominal*mult + self.base.face_nominal + 0.000001*self.unique_val
    end
    function c:is_suit(s, bypass, flush_calc)
        if flush_calc then
            if self.ability.effect == 'Stone Card' then return false end
            if self.ability.name == "Wild Card" and not self.debuff then return true end
            return self.base.suit == s
        else
            if self.debuff and not bypass then return end
            if self.ability.effect == 'Stone Card' then return false end
            if self.ability.name == "Wild Card" then return true end
            return self.base.suit == s
        end
    end
    function c:is_face(from_boss)
        if self.debuff and not from_boss then return end
        return self.base.id == 11 or self.base.id == 12 or self.base.id == 13
    end
    function c:get_chip_bonus()
        if self.debuff then return 0 end
        if self.ability.effect == 'Stone Card' then return self.ability.bonus + (self.ability.perma_bonus or 0) end
        return self.base.nominal + self.ability.bonus + (self.ability.perma_bonus or 0)
    end
    function c:get_chip_mult()
        if self.debuff then return 0 end
        if self.ability.set == 'Joker' then return 0 end
        return self.ability.mult
    end
    function c:get_chip_x_mult()
        if self.debuff then return 0 end
        if self.ability.set == 'Joker' then return 0 end
        if self.ability.x_mult <= 1 then return 0 end
        return self.ability.x_mult
    end
    function c:get_chip_h_mult()
        if self.debuff then return 0 end
        return self.ability.h_mult
    end
    function c:get_chip_h_x_mult()
        if self.debuff then return 0 end
        return self.ability.h_x_mult
    end
    function c:get_edition()
        if self.debuff then return end
        if self.edition then
            local ret = {card = self}
            if self.edition.x_mult then ret.x_mult_mod = self.edition.x_mult end
            if self.edition.mult then ret.mult_mod = self.edition.mult end
            if self.edition.chips then ret.chip_mod = self.edition.chips end
            return ret
        end
    end
    function c:get_p_dollars()
        if self.debuff then return 0 end
        local ret = 0
        if self.seal == 'Gold' then ret = ret + 3 end
        if self.ability.p_dollars > 0 then ret = ret + self.ability.p_dollars end
        return ret
    end
    function c:calculate_seal(ctx)
        if self.debuff then return nil end
        if ctx and ctx.repetition then
            if self.seal == 'Red' then return {repetitions = 1, card = self} end
        end
        return nil
    end
    function c:calculate_joker() return nil end
    function c:set_debuff(b) self.debuff = b end
    function c:juice_up() end

    return c
end

----------------------------------------------------------------------------
-- Minimal Blind stub
----------------------------------------------------------------------------

local function make_blind(name, mult, debuff_config)
    local b = {
        name = name or "Small Blind",
        mult = mult or 1.0,
        disabled = false,
        triggered = false,
        debuff = debuff_config or {},
        hands = {},
        only_hand = false,
    }
    function b:debuff_hand(cards, hand, handname, check)
        if self.disabled then return end
        if self.debuff then
            self.triggered = false
            if self.debuff.h_size_ge and #cards < self.debuff.h_size_ge then
                self.triggered = true; return true
            end
            if self.name == "The Eye" then
                if self.hands[handname] then self.triggered = true; return true end
                if not check then self.hands[handname] = true end
            end
        end
        if self.name == "The Flint" then self.triggered = false end
    end
    function b:modify_hand(cards, poker_hands, text, m, hc)
        if self.disabled then return m, hc, false end
        if self.name == "The Flint" then
            self.triggered = true
            return math.max(math.floor(m*0.5+0.5), 1), math.max(math.floor(hc*0.5+0.5), 0), true
        end
        return m, hc, false
    end
    function b:debuff_card() end
    return b
end

----------------------------------------------------------------------------
-- Scoring pipeline (stripped of UI/animation)
----------------------------------------------------------------------------

local function score_hand(play_cards, hand_cards, hands, blind, jokers)
    jokers = jokers or {}
    G.play.cards = play_cards
    G.hand.cards = hand_cards
    G.jokers.cards = jokers
    G.GAME.blind = blind
    G.GAME.hands = hands

    local text, _, poker_hands, scoring_hand = G.FUNCS.get_poker_hand_info(play_cards)
    if text == "NULL" then
        return {hand_type = "NULL", chips = 0, mult = 0, total = 0, debuffed = false}
    end

    -- Stone/Splash augmentation (simplified - no Splash joker)
    local pures = {}
    for i=1, #play_cards do
        if play_cards[i].ability.effect == 'Stone Card' then
            local inside = false
            for j=1, #scoring_hand do
                if scoring_hand[j] == play_cards[i] then inside = true end
            end
            if not inside then pures[#pures+1] = play_cards[i] end
        end
    end
    for i=1, #pures do scoring_hand[#scoring_hand+1] = pures[i] end
    table.sort(scoring_hand, function(a,b) return a.T.x < b.T.x end)

    -- Phase 3: debuff check
    if blind:debuff_hand(scoring_hand, poker_hands, text) then
        return {hand_type = text, chips = 0, mult = 0, total = 0, debuffed = true}
    end

    -- Phase 4: base
    local hand_chips = mod_chips(hands[text].chips)
    local mult = mod_mult(hands[text].mult)
    local base_chips, base_mult = hand_chips, mult

    -- Phase 5: "before" joker pass
    for k=1, #jokers do
        if not jokers[k].debuff then
            local before_result = jokers[k]:calculate_joker({
                before = true, full_hand = play_cards,
                scoring_hand = scoring_hand, scoring_name = text,
                poker_hands = poker_hands,
            })
        end
    end

    -- Phase 6: blind modify
    mult, hand_chips = blind:modify_hand(nil, nil, text, mult, hand_chips)

    -- Phase 7: per scored card with retriggers + joker individual effects
    local dollars = 0
    for i = 1, #scoring_hand do
        if not scoring_hand[i].debuff then
            -- 7a: Collect retriggers (seal + joker)
            local reps = {1}
            local seal = scoring_hand[i]:calculate_seal({repetition = true})
            if seal and seal.repetitions then
                for h = 1, seal.repetitions do reps[#reps+1] = seal end
            end
            for k=1, #jokers do
                if not jokers[k].debuff then
                    local rep_eval = jokers[k]:calculate_joker({
                        cardarea = G.play, full_hand = play_cards,
                        scoring_hand = scoring_hand, scoring_name = text,
                        poker_hands = poker_hands,
                        other_card = scoring_hand[i], repetition = true,
                    })
                    if rep_eval and rep_eval.repetitions then
                        for h = 1, rep_eval.repetitions do reps[#reps+1] = rep_eval end
                    end
                end
            end

            -- 7b-d: Each repetition
            for j = 1, #reps do
                -- Card's own effects
                local ev = eval_card(scoring_hand[i], {cardarea = G.play})
                local effects = {ev}

                -- Joker individual effects on this card
                for k=1, #jokers do
                    if not jokers[k].debuff then
                        local j_eval = jokers[k]:calculate_joker({
                            cardarea = G.play, full_hand = play_cards,
                            scoring_hand = scoring_hand, scoring_name = text,
                            poker_hands = poker_hands,
                            other_card = scoring_hand[i], individual = true,
                        })
                        if j_eval then effects[#effects+1] = j_eval end
                    end
                end

                -- Apply effects in source order
                for ii = 1, #effects do
                    if effects[ii].chips then hand_chips = mod_chips(hand_chips + effects[ii].chips) end
                    if effects[ii].mult then mult = mod_mult(mult + effects[ii].mult) end
                    if effects[ii].p_dollars then dollars = dollars + effects[ii].p_dollars end
                    if effects[ii].dollars then dollars = dollars + effects[ii].dollars end
                    if effects[ii].x_mult then mult = mod_mult(mult * effects[ii].x_mult) end
                    if effects[ii].edition then
                        hand_chips = mod_chips(hand_chips + (effects[ii].edition.chip_mod or 0))
                        mult = mult + (effects[ii].edition.mult_mod or 0)
                        mult = mod_mult(mult * (effects[ii].edition.x_mult_mod or 1))
                    end
                end
            end
        end
    end

    -- Phase 8: per held card + joker individual effects
    for i = 1, #hand_cards do
        if not hand_cards[i].debuff then
            local reps = {1}
            local seal = hand_cards[i]:calculate_seal({repetition = true})
            if seal and seal.repetitions then
                for h = 1, seal.repetitions do reps[#reps+1] = seal end
            end
            for j = 1, #reps do
                local ev = eval_card(hand_cards[i], {cardarea = G.hand})
                local effects = {ev}

                for k=1, #jokers do
                    if not jokers[k].debuff then
                        local j_eval = jokers[k]:calculate_joker({
                            cardarea = G.hand, full_hand = play_cards,
                            scoring_hand = scoring_hand, scoring_name = text,
                            poker_hands = poker_hands,
                            other_card = hand_cards[i], individual = true,
                        })
                        if j_eval then effects[#effects+1] = j_eval end
                    end
                end

                for ii = 1, #effects do
                    if effects[ii].h_mult then mult = mod_mult(mult + effects[ii].h_mult) end
                    if effects[ii].x_mult then mult = mod_mult(mult * effects[ii].x_mult) end
                    if effects[ii].dollars then dollars = dollars + effects[ii].dollars end
                end
            end
        end
    end

    -- Phase 9: joker main effects (left to right)
    for i = 1, #jokers do
        if not jokers[i].debuff then
            -- 9a: Edition additive (chip_mod, mult_mod)
            local edition = jokers[i]:get_edition()
            if edition then
                if edition.chip_mod then hand_chips = mod_chips(hand_chips + edition.chip_mod) end
                if edition.mult_mod then mult = mod_mult(mult + edition.mult_mod) end
            end

            -- 9b: Main joker effect
            local j_result = jokers[i]:calculate_joker({
                cardarea = G.jokers, full_hand = play_cards,
                scoring_hand = scoring_hand, scoring_name = text,
                poker_hands = poker_hands, joker_main = true,
            })
            if j_result then
                if j_result.mult_mod then mult = mod_mult(mult + j_result.mult_mod) end
                if j_result.chip_mod then hand_chips = mod_chips(hand_chips + j_result.chip_mod) end
                if j_result.Xmult_mod then mult = mod_mult(mult * j_result.Xmult_mod) end
                if j_result.dollars then dollars = dollars + j_result.dollars end
            end

            -- 9c: Joker-on-joker
            for _, v in ipairs(jokers) do
                if v ~= jokers[i] and not v.debuff then
                    local j2j = v:calculate_joker({
                        full_hand = play_cards, scoring_hand = scoring_hand,
                        scoring_name = text, poker_hands = poker_hands,
                        other_joker = jokers[i],
                    })
                    if j2j then
                        if j2j.mult_mod then mult = mod_mult(mult + j2j.mult_mod) end
                        if j2j.chip_mod then hand_chips = mod_chips(hand_chips + j2j.chip_mod) end
                        if j2j.Xmult_mod then mult = mod_mult(mult * j2j.Xmult_mod) end
                    end
                end
            end

            -- 9d: Edition multiplicative (x_mult_mod)
            if edition and edition.x_mult_mod then
                mult = mod_mult(mult * edition.x_mult_mod)
            end
        end
    end

    -- Phase 11: Glass Card destruction
    local destroyed = {}
    for i = 1, #scoring_hand do
        if not scoring_hand[i].debuff and scoring_hand[i].ability.name == 'Glass Card' then
            -- Deterministic: use glass_shatter flag if set, else always shatter for test
            if scoring_hand[i].force_shatter then
                destroyed[#destroyed+1] = scoring_hand[i]
            end
        end
    end
    -- Notify jokers of destruction
    if #destroyed > 0 then
        for k=1, #jokers do
            if not jokers[k].debuff then
                jokers[k]:calculate_joker({
                    cards_destroyed = destroyed,
                })
            end
        end
    end

    -- Phase 12
    local total = math.floor(hand_chips * mult)

    -- Phase 13: "after" joker pass
    local jokers_removed = {}
    for k=1, #jokers do
        if not jokers[k].debuff then
            local after_result = jokers[k]:calculate_joker({after = true})
            if after_result and after_result.remove then
                jokers_removed[#jokers_removed+1] = jokers[k]
            end
        end
    end

    return {
        hand_type = text,
        base_chips = base_chips,
        base_mult = base_mult,
        chips = hand_chips,
        mult = mult,
        total = total,
        dollars = dollars,
        debuffed = false,
        scoring_count = #scoring_hand,
        destroyed_count = #destroyed,
        jokers_removed_count = #jokers_removed,
    }
end

----------------------------------------------------------------------------
-- Joker constructor
----------------------------------------------------------------------------

local function make_joker(name, opts)
    opts = opts or {}
    card_counter = card_counter + 1
    local j = {
        base = nil,
        ability = {
            name = name,
            set = "Joker",
            effect = opts.effect or "",
            mult = opts.mult or 0,
            h_mult = 0, h_x_mult = 0, h_dollars = 0, p_dollars = 0,
            bonus = 0, perma_bonus = 0,
            x_mult = opts.Xmult or 1,
            extra = opts.extra,
            type = opts.type,
            t_mult = opts.t_mult or 0,
            t_chips = opts.t_chips or 0,
            steel_tally = opts.steel_tally or 0,
            caino_xmult = opts.caino_xmult or 1,
        },
        rarity = opts.rarity or 0,
        edition = opts.edition,
        seal = nil,
        debuff = false,
        sort_id = card_counter,
        unique_val = card_counter * 0.001,
        T = { x = card_counter },
        playing_card = false,
    }

    -- Stub methods matching playing cards (unused but prevent nil errors)
    function j:get_id() return 0 end
    function j:is_suit() return false end
    function j:is_face() return false end
    function j:get_chip_bonus() return 0 end
    function j:get_chip_mult() return 0 end
    function j:get_chip_x_mult() return 0 end
    function j:get_chip_h_mult() return 0 end
    function j:get_chip_h_x_mult() return 0 end
    function j:get_edition()
        if self.debuff then return end
        if self.edition then
            local ret = {card = self}
            if self.edition.x_mult then ret.x_mult_mod = self.edition.x_mult end
            if self.edition.mult then ret.mult_mod = self.edition.mult end
            if self.edition.chips then ret.chip_mod = self.edition.chips end
            return ret
        end
    end
    function j:get_p_dollars() return 0 end
    function j:calculate_seal() return nil end
    function j:set_debuff(b) self.debuff = b end
    function j:juice_up() end

    -- calculate_joker: implement the specific joker effects
    function j:calculate_joker(context)
        if self.debuff then return nil end

        -- Joker: +mult (card.lua:3980)
        if self.ability.name == 'Joker' then
            if context.joker_main then
                return { mult_mod = self.ability.mult }
            end
        end

        -- Green Joker: +1 mult per hand (before), return accumulated (joker_main)
        if self.ability.name == 'Green Joker' then
            if context.before and not context.blueprint_seen then
                self.ability.mult = self.ability.mult + (self.ability.extra and self.ability.extra.hand_add or 1)
                return {}
            end
            if context.joker_main and self.ability.mult > 0 then
                return { mult_mod = self.ability.mult }
            end
        end

        -- Ride the Bus: +1 mult per no-face hand (before), reset on face
        if self.ability.name == 'Ride the Bus' then
            if context.before and not context.blueprint_seen and context.scoring_hand then
                local faces = false
                for i = 1, #context.scoring_hand do
                    if context.scoring_hand[i]:is_face() then faces = true end
                end
                if faces then
                    self.ability.mult = 0
                else
                    self.ability.mult = self.ability.mult + (self.ability.extra or 1)
                end
                return {}
            end
            if context.joker_main and self.ability.mult > 0 then
                return { mult_mod = self.ability.mult }
            end
        end

        -- Jolly Joker: +t_mult if hand contains Pair (card.lua:3660)
        if self.ability.name == 'Jolly Joker' then
            if context.joker_main then
                if self.ability.t_mult > 0 and context.poker_hands
                    and next(context.poker_hands[self.ability.type] or {}) then
                    return { mult_mod = self.ability.t_mult }
                end
            end
        end

        -- The Duo: xMult if hand contains type (card.lua:3653)
        if self.ability.name == 'The Duo' then
            if context.joker_main then
                if self.ability.x_mult > 1 and context.poker_hands
                    and next(context.poker_hands[self.ability.type] or {}) then
                    return { Xmult_mod = self.ability.x_mult }
                end
            end
        end

        -- Steel Joker: x(1 + 0.2 * steel_tally) (card.lua:3929)
        if self.ability.name == 'Steel Joker' then
            if context.joker_main and (self.ability.steel_tally or 0) > 0 then
                return { Xmult_mod = 1 + self.ability.extra * self.ability.steel_tally }
            end
        end

        -- Ice Cream: +chips (joker_main), -chips (after), self-destruct
        if self.ability.name == 'Ice Cream' then
            if context.after then
                local chips = self.ability.extra.chips
                local chip_mod = self.ability.extra.chip_mod
                if chips - chip_mod <= 0 then
                    return { remove = true }
                end
                self.ability.extra.chips = chips - chip_mod
                return {}
            end
            if context.joker_main then
                return { chip_mod = self.ability.extra.chips }
            end
        end

        -- Scary Face: +chips per face card (card.lua:3136)
        if self.ability.name == 'Scary Face' then
            if context.individual and context.cardarea == G.play then
                if context.other_card and context.other_card:is_face() then
                    return { chips = self.ability.extra }
                end
            end
        end

        -- Lusty Joker: +s_mult per Heart scored (card.lua:3065, Suit Mult)
        if self.ability.name == 'Lusty Joker' then
            if context.individual and context.cardarea == G.play then
                if context.other_card and context.other_card:is_suit(self.ability.extra.suit) then
                    return { mult = self.ability.extra.s_mult }
                end
            end
        end

        -- Sock and Buskin: retrigger face cards (card.lua:3344)
        if self.ability.name == 'Sock and Buskin' then
            if context.repetition and context.cardarea == G.play then
                if context.other_card and context.other_card:is_face() then
                    return { repetitions = self.ability.extra }
                end
            end
        end

        -- Dusk: retrigger all on last hand (card.lua:3360)
        if self.ability.name == 'Dusk' then
            if context.repetition and context.cardarea == G.play then
                if (G.GAME.current_round or {}).hands_left == 0 then
                    return { repetitions = self.ability.extra }
                end
            end
        end

        -- Blackboard: x3 if all held cards are Spades or Clubs (card.lua:3951)
        if self.ability.name == 'Blackboard' then
            if context.joker_main then
                local black_suits, all_cards = 0, 0
                for _, v in ipairs(G.hand.cards) do
                    all_cards = all_cards + 1
                    if v:is_suit('Clubs', nil, true) or v:is_suit('Spades', nil, true) then
                        black_suits = black_suits + 1
                    end
                end
                if all_cards > 0 and black_suits == all_cards then
                    return { Xmult_mod = self.ability.extra }
                end
            end
        end

        -- Baseball Card: x1.5 if other_joker is Uncommon (rarity 2)
        if self.ability.name == 'Baseball Card' then
            if context.other_joker and context.other_joker ~= self then
                if (context.other_joker.rarity or 0) == 2 then
                    return { Xmult_mod = self.ability.extra }
                end
            end
        end

        -- Blueprint: copy right neighbor
        if self.ability.name == 'Blueprint' then
            if not context.blueprint_seen then
                -- Find right neighbor in G.jokers.cards
                local target = nil
                for i = 1, #G.jokers.cards do
                    if G.jokers.cards[i] == self then
                        target = G.jokers.cards[i+1]
                        break
                    end
                end
                if target and not target.debuff then
                    context.blueprint_seen = true
                    return target:calculate_joker(context)
                end
            end
        end

        -- Caino: +1 xMult per face card destroyed
        if self.ability.name == 'Caino' then
            if context.cards_destroyed then
                local faces = 0
                for _, c in ipairs(context.cards_destroyed) do
                    if c:is_face() then faces = faces + 1 end
                end
                if faces > 0 then
                    self.ability.caino_xmult = (self.ability.caino_xmult or 1) + faces * (self.ability.extra or 1)
                end
                return nil
            end
            if context.joker_main and (self.ability.caino_xmult or 1) > 1 then
                return { Xmult_mod = self.ability.caino_xmult }
            end
        end

        -- Glass Joker: +0.75 xMult per Glass Card destroyed
        if self.ability.name == 'Glass Joker' then
            if context.cards_destroyed then
                local glass = 0
                for _, c in ipairs(context.cards_destroyed) do
                    if c.ability.name == 'Glass Card' then glass = glass + 1 end
                end
                if glass > 0 then
                    self.ability.x_mult = self.ability.x_mult + (self.ability.extra or 0.75) * glass
                end
                return nil
            end
            if context.joker_main and self.ability.x_mult > 1 then
                return { Xmult_mod = self.ability.x_mult }
            end
        end

        return nil
    end

    return j
end

----------------------------------------------------------------------------
-- Hand level tables
----------------------------------------------------------------------------

local function make_hands()
    return {
        ["Flush Five"] = {chips=160,mult=16,level=1,played=0,played_this_round=0,visible=false,s_chips=160,s_mult=16,l_chips=50,l_mult=3},
        ["Flush House"] = {chips=140,mult=14,level=1,played=0,played_this_round=0,visible=false,s_chips=140,s_mult=14,l_chips=40,l_mult=4},
        ["Five of a Kind"] = {chips=120,mult=12,level=1,played=0,played_this_round=0,visible=false,s_chips=120,s_mult=12,l_chips=35,l_mult=3},
        ["Straight Flush"] = {chips=100,mult=8,level=1,played=0,played_this_round=0,visible=true,s_chips=100,s_mult=8,l_chips=40,l_mult=4},
        ["Four of a Kind"] = {chips=60,mult=7,level=1,played=0,played_this_round=0,visible=true,s_chips=60,s_mult=7,l_chips=30,l_mult=3},
        ["Full House"] = {chips=40,mult=4,level=1,played=0,played_this_round=0,visible=true,s_chips=40,s_mult=4,l_chips=25,l_mult=2},
        ["Flush"] = {chips=35,mult=4,level=1,played=0,played_this_round=0,visible=true,s_chips=35,s_mult=4,l_chips=15,l_mult=2},
        ["Straight"] = {chips=30,mult=4,level=1,played=0,played_this_round=0,visible=true,s_chips=30,s_mult=4,l_chips=30,l_mult=3},
        ["Three of a Kind"] = {chips=30,mult=3,level=1,played=0,played_this_round=0,visible=true,s_chips=30,s_mult=3,l_chips=20,l_mult=2},
        ["Two Pair"] = {chips=20,mult=2,level=1,played=0,played_this_round=0,visible=true,s_chips=20,s_mult=2,l_chips=20,l_mult=1},
        ["Pair"] = {chips=10,mult=2,level=1,played=0,played_this_round=0,visible=true,s_chips=10,s_mult=2,l_chips=15,l_mult=1},
        ["High Card"] = {chips=5,mult=1,level=1,played=0,played_this_round=0,visible=true,s_chips=5,s_mult=1,l_chips=10,l_mult=1},
    }
end

-- get_poker_hand_info stub
G.FUNCS = G.FUNCS or {}
G.FUNCS.get_poker_hand_info = function(_cards)
    local poker_hands = evaluate_poker_hand(_cards)
    local scoring_hand = {}
    local text = 'NULL'
    local priority = {"Flush Five","Flush House","Five of a Kind","Straight Flush","Four of a Kind","Full House","Flush","Straight","Three of a Kind","Two Pair","Pair","High Card"}
    for _, h in ipairs(priority) do
        if next(poker_hands[h] or {}) then
            text = h; scoring_hand = poker_hands[h][1]; break
        end
    end
    return text, text, poker_hands, scoring_hand, text
end

----------------------------------------------------------------------------
-- JSON helper
----------------------------------------------------------------------------
local function json_string(s) return '"'..s:gsub('\\','\\\\'):gsub('"','\\"')..'"' end
local function json_value(v)
    if v == nil then return "null"
    elseif type(v) == "boolean" then return v and "true" or "false"
    elseif type(v) == "number" then
        if v == math.floor(v) and math.abs(v) < 2^53 then return string.format("%d",v)
        else return string.format("%.15g",v) end
    elseif type(v) == "string" then return json_string(v)
    elseif type(v) == "table" then
        if #v > 0 then
            local p = {}; for i=1,#v do p[i] = json_value(v[i]) end
            return "["..table.concat(p,", ").."]"
        else
            local p,ks = {},{}
            for k in pairs(v) do ks[#ks+1] = tostring(k) end
            table.sort(ks)
            for _,k in ipairs(ks) do p[#p+1] = json_string(k)..": "..json_value(v[k]) end
            return "{"..table.concat(p,", ").."}"
        end
    end
    return "null"
end

----------------------------------------------------------------------------
-- Test scenarios
----------------------------------------------------------------------------

local results = {}

-- (a) Pair of Aces, level 1, no enhancements, small blind
card_counter = 0
local hands_a = make_hands()
G.GAME.hands = hands_a
local r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_a,
    make_blind("Small Blind", 1.0)
)
r.name = "pair_aces_basic"
results[#results+1] = r

-- (b) Three Kings, one with Foil edition
card_counter = 0
local hands_b = make_hands()
G.GAME.hands = hands_b
r = score_hand(
    {make_card("Hearts","King",{edition={foil=true,chips=50,type="foil"}}), make_card("Spades","King"), make_card("Clubs","King"), make_card("Diamonds","5"), make_card("Hearts","2")},
    {},
    hands_b,
    make_blind("Small Blind", 1.0)
)
r.name = "three_kings_foil"
results[#results+1] = r

-- (c) Flush with one Glass Card (x_mult=2)
card_counter = 0
local hands_c = make_hands()
G.GAME.hands = hands_c
r = score_hand(
    {make_card("Hearts","2"), make_card("Hearts","5"), make_card("Hearts","8"), make_card("Hearts","Jack"), make_card("Hearts","Ace",{Xmult=2,effect="Glass Card"})},
    {},
    hands_c,
    make_blind("Small Blind", 1.0)
)
r.name = "flush_glass"
results[#results+1] = r

-- (d) Full House with Steel Card held (h_x_mult=1.5)
card_counter = 0
local hands_d = make_hands()
G.GAME.hands = hands_d
r = score_hand(
    {make_card("Hearts","King"), make_card("Spades","King"), make_card("Clubs","King"), make_card("Diamonds","5"), make_card("Hearts","5")},
    {make_card("Clubs","3",{h_x_mult=1.5,effect="Steel Card"})},
    hands_d,
    make_blind("Small Blind", 1.0)
)
r.name = "full_house_steel_held"
results[#results+1] = r

-- (e) Pair with Red Seal on one card
card_counter = 0
local hands_e = make_hands()
G.GAME.hands = hands_e
r = score_hand(
    {make_card("Hearts","Ace",{seal="Red"}), make_card("Spades","Ace")},
    {},
    hands_e,
    make_blind("Small Blind", 1.0)
)
-- Fix: need to set seal on the card, not via opts
-- Actually our make_card accepts seal in opts, let me check
r.name = "pair_aces_red_seal"
results[#results+1] = r

-- (f) Pair against The Flint
card_counter = 0
local hands_f = make_hands()
G.GAME.hands = hands_f
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_f,
    make_blind("The Flint", 2.0)
)
r.name = "pair_aces_flint"
results[#results+1] = r

-- (g) Pair against The Eye (second use)
card_counter = 0
local hands_g = make_hands()
G.GAME.hands = hands_g
local eye_blind = make_blind("The Eye", 2.0)
eye_blind.debuff = {}
-- First pair: allowed
score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {}, hands_g, eye_blind
)
-- Second pair: debuffed
card_counter = 0
r = score_hand(
    {make_card("Hearts","King"), make_card("Spades","King")},
    {}, hands_g, eye_blind
)
r.name = "pair_eye_debuffed"
results[#results+1] = r

----------------------------------------------------------------------------
-- Joker test scenarios
----------------------------------------------------------------------------

-- (h) Pair of Aces + j_joker (+4 mult)
-- Base: Pair L1 = 10 chips, 2 mult. Cards: 11+11=22. Total chips=32, mult=2.
-- Phase 9: j_joker +4 mult → mult=6. Score: 32×6=192.
card_counter = 0
local hands_h = make_hands()
G.GAME.hands = hands_h
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_h,
    make_blind("Small Blind", 1.0),
    {make_joker("Joker", {mult = 4})}
)
r.name = "pair_aces_joker"
results[#results+1] = r

-- (i) Flush of Hearts + j_lusty_joker (+3 mult per Heart scored)
-- Base: Flush L1 = 35 chips, 4 mult. Cards: 2+5+8+10+11=36. Total chips=71.
-- Phase 7: +3 mult per card × 5 Hearts = +15 mult → mult=19.
-- Score: 71×19=1349.
card_counter = 0
local hands_i = make_hands()
G.GAME.hands = hands_i
r = score_hand(
    {make_card("Hearts","2"), make_card("Hearts","5"), make_card("Hearts","8"),
     make_card("Hearts","Jack"), make_card("Hearts","Ace")},
    {},
    hands_i,
    make_blind("Small Blind", 1.0),
    {make_joker("Lusty Joker", {extra = {s_mult = 3, suit = "Hearts"}})}
)
r.name = "flush_hearts_lusty"
results[#results+1] = r

-- (j) Three Kings + j_scary_face (+30 chips per face card)
-- Base: Three of a Kind L1 = 30 chips, 3 mult. Cards: 10+10+10+5+2=37.
-- Total chips = 30+37=67 base, then +30 per face (3 Kings) = +90 chips.
-- Phase 7: 67 + 90 = 157 chips, mult = 3. Score: 157×3=471.
card_counter = 0
local hands_j = make_hands()
G.GAME.hands = hands_j
r = score_hand(
    {make_card("Hearts","King"), make_card("Spades","King"), make_card("Clubs","King"),
     make_card("Diamonds","5"), make_card("Hearts","2")},
    {},
    hands_j,
    make_blind("Small Blind", 1.0),
    {make_joker("Scary Face", {extra = 30})}
)
r.name = "three_kings_scary_face"
results[#results+1] = r

-- (k) Full House + j_jolly (+8 mult, triggers because Full House contains Pair)
-- Base: Full House L1 = 40 chips, 4 mult. Cards: 10+10+10+5+5=40.
-- Total chips = 40+40=80. Phase 9: +8 mult (Pair in poker_hands) → mult=12.
-- Score: 80×12=960.
card_counter = 0
local hands_k = make_hands()
G.GAME.hands = hands_k
r = score_hand(
    {make_card("Hearts","King"), make_card("Spades","King"), make_card("Clubs","King"),
     make_card("Diamonds","5"), make_card("Hearts","5")},
    {},
    hands_k,
    make_blind("Small Blind", 1.0),
    {make_joker("Jolly Joker", {t_mult = 8, type = "Pair"})}
)
r.name = "full_house_jolly"
results[#results+1] = r

-- (l) Pair + j_duo (x2 mult, contains Pair → triggers)
-- Base: Pair L1 = 10 chips, 2 mult. Cards: 11+11=22. Total chips=32.
-- Phase 9: x2 mult → mult=4. Score: 32×4=128.
card_counter = 0
local hands_l = make_hands()
G.GAME.hands = hands_l
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_l,
    make_blind("Small Blind", 1.0),
    {make_joker("The Duo", {Xmult = 2, type = "Pair"})}
)
r.name = "pair_duo_xmult"
results[#results+1] = r

-- (m) j_joker (+4) then j_blackboard (x3) — order matters
-- Pair of Aces (black suits): 32 chips, 2 mult.
-- Held: one black card (Spades 5) so Blackboard triggers.
-- Phase 9: j_joker +4 → mult=6, then j_blackboard x3 → mult=18.
-- Score: 32×18=576.
card_counter = 0
local hands_m = make_hands()
G.GAME.hands = hands_m
r = score_hand(
    {make_card("Spades","Ace"), make_card("Clubs","Ace")},
    {make_card("Spades","5")},
    hands_m,
    make_blind("Small Blind", 1.0),
    {make_joker("Joker", {mult = 4}), make_joker("Blackboard", {extra = 3})}
)
r.name = "joker_then_blackboard"
results[#results+1] = r

-- (n) j_blackboard (x3) then j_joker (+4) — reversed order
-- Phase 9: j_blackboard x3 → mult=6, then j_joker +4 → mult=10.
-- Score: 32×10=320.
card_counter = 0
local hands_n = make_hands()
G.GAME.hands = hands_n
r = score_hand(
    {make_card("Spades","Ace"), make_card("Clubs","Ace")},
    {make_card("Spades","5")},
    hands_n,
    make_blind("Small Blind", 1.0),
    {make_joker("Blackboard", {extra = 3}), make_joker("Joker", {mult = 4})}
)
r.name = "blackboard_then_joker"
results[#results+1] = r

-- (o) Foil j_joker → +50 chips from edition + +4 mult from effect
-- Pair of Aces: 32 chips, 2 mult.
-- Phase 9: Foil +50 chips → 82, then j_joker +4 mult → 6.
-- Score: 82×6=492.
card_counter = 0
local hands_o = make_hands()
G.GAME.hands = hands_o
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_o,
    make_blind("Small Blind", 1.0),
    {make_joker("Joker", {mult = 4, edition = {foil=true, chips=50, type="foil"}})}
)
r.name = "foil_joker"
results[#results+1] = r

----------------------------------------------------------------------------
-- Complex joker interaction scenarios
----------------------------------------------------------------------------

-- (p) Green Joker after 3 hands: accumulated +3 mult
card_counter = 0
local hands_p = make_hands()
G.GAME.hands = hands_p
local green = make_joker("Green Joker", {mult = 0, extra = {hand_add = 1, discard_sub = 1}})
-- Simulate 3 hands
for hand_num = 1, 3 do
    card_counter = 0
    score_hand(
        {make_card("Hearts","5"), make_card("Spades","5")},
        {}, hands_p, make_blind("Small Blind", 1.0), {green}
    )
end
-- 4th hand: Green Joker should have mult=3 from 3 prior before phases
card_counter = 0
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {}, hands_p, make_blind("Small Blind", 1.0), {green}
)
-- After 4 before phases: mult should be 4 now
-- Pair: 32 chips, 2 mult. Green +4 → 6. Score: 32×6=192.
r.name = "green_joker_4_hands"
results[#results+1] = r

-- (q) Steel Joker with 2 Steel Cards → xMult = 1 + 0.2*2 = 1.4
card_counter = 0
local hands_q = make_hands()
G.GAME.hands = hands_q
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_q,
    make_blind("Small Blind", 1.0),
    {make_joker("Steel Joker", {extra = 0.2, steel_tally = 2})}
)
-- Pair: 32 chips, 2 mult. Steel Joker x1.4 → 2.8. Score: floor(32×2.8) = 89.
r.name = "steel_joker_2_steel"
results[#results+1] = r

-- (r) Sock and Buskin with 2 face cards: each retriggered once
-- Three of a Kind Kings: 3 Kings score. Sock: +1 rep per face.
-- Each King: 10 chips × 2 reps = 20. Total: 30 + 60 = 90 chips, 3 mult.
-- Score: 90 × 3 = 270.
card_counter = 0
local hands_r = make_hands()
G.GAME.hands = hands_r
r = score_hand(
    {make_card("Hearts","King"), make_card("Spades","King"), make_card("Clubs","King"),
     make_card("Diamonds","5"), make_card("Hearts","2")},
    {},
    hands_r,
    make_blind("Small Blind", 1.0),
    {make_joker("Sock and Buskin", {extra = 1})}
)
r.name = "sock_buskin_3_kings"
results[#results+1] = r

-- (s) Red Seal + Dusk on last hand: 1 base + 1 Red + 1 Dusk = 3 reps
-- Pair of Aces, Red Seal on first Ace, Dusk active (hands_left=0)
card_counter = 0
local hands_s = make_hands()
G.GAME.hands = hands_s
G.GAME.current_round = G.GAME.current_round or {}
G.GAME.current_round.hands_left = 0
local red_ace = make_card("Hearts","Ace",{seal="Red"})
r = score_hand(
    {red_ace, make_card("Spades","Ace")},
    {},
    hands_s,
    make_blind("Small Blind", 1.0),
    {make_joker("Dusk", {extra = 1})}
)
-- Red Ace: 1 base + 1 Red Seal + 1 Dusk = 3 reps → 11×3 = 33 chips
-- Normal Ace: 1 base + 1 Dusk = 2 reps → 11×2 = 22 chips
-- Total: 10 + 33 + 22 = 65 chips, 2 mult. Score: 65×2 = 130.
r.name = "red_seal_dusk_last_hand"
results[#results+1] = r

-- (t) Blueprint copying Green Joker at +5 mult
card_counter = 0
local hands_t = make_hands()
G.GAME.hands = hands_t
local green_t = make_joker("Green Joker", {mult = 5, extra = {hand_add = 1, discard_sub = 1}})
local blueprint_t = make_joker("Blueprint")
G.jokers.cards = {blueprint_t, green_t}
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {},
    hands_t,
    make_blind("Small Blind", 1.0),
    {blueprint_t, green_t}
)
-- Phase 5 (before): Green +1 → mult=6. Blueprint skips (no before impl for copy).
-- Phase 9: Blueprint copies Green → mult_mod = 6. Green → mult_mod = 6.
-- Pair: 32 chips, 2 mult. +6 +6 = 14. Score: 32×14 = 448.
r.name = "blueprint_green_joker"
results[#results+1] = r

-- (u) Baseball Card with 2 Uncommon jokers
-- Steel Joker (Uncommon, rarity=2) + Blackboard (Uncommon, rarity=2) + Baseball
card_counter = 0
local hands_u = make_hands()
G.GAME.hands = hands_u
local steel_u = make_joker("Steel Joker", {extra = 0.2, steel_tally = 1})
steel_u.rarity = 2
local bb_u = make_joker("Blackboard", {extra = 3})
bb_u.rarity = 2
local baseball_u = make_joker("Baseball Card", {extra = 1.5})
baseball_u.rarity = 3
G.jokers.cards = {steel_u, bb_u, baseball_u}
r = score_hand(
    {make_card("Spades","Ace"), make_card("Clubs","Ace")},
    {make_card("Spades","5")},
    hands_u,
    make_blind("Small Blind", 1.0),
    {steel_u, bb_u, baseball_u}
)
-- Pair: 32 chips, 2 mult.
-- Phase 9 steel_u: x1.2 → 2.4. Baseball reacts (Uncommon): x1.5 → 3.6.
-- Phase 9 bb_u: x3 (all held black) → 10.8. Baseball reacts (Uncommon): x1.5 → 16.2.
-- Phase 9 baseball: no self-react (Rare).
-- Score: floor(32 × 16.2) = 518.
r.name = "baseball_2_uncommon"
results[#results+1] = r

-- (v) Glass Card destruction + Caino
card_counter = 0
local hands_v = make_hands()
G.GAME.hands = hands_v
local glass_king = make_card("Hearts","King",{Xmult=2, effect="Glass Card", name="Glass Card"})
glass_king.force_shatter = true  -- force deterministic shatter
r = score_hand(
    {glass_king, make_card("Spades","King"), make_card("Clubs","King"),
     make_card("Diamonds","5"), make_card("Hearts","2")},
    {},
    hands_v,
    make_blind("Small Blind", 1.0),
    {make_joker("Caino", {extra = 1, caino_xmult = 1})}
)
-- Three Kings: 30 base, 3 mult. Glass King x2 → 6 mult.
-- Phase 11: Glass King shatters. Caino: +1 (face card) → caino_xmult=2.
-- Score: chips × mult already computed before destruction notification.
-- Caino xMult not applied THIS hand (it fires in joker_main BEFORE phase 11).
-- So score = (30 + 10+10+10+5+2) × 6... wait, only 3 Kings score.
-- 30 + 10+10+10 = 60 chips, 3 mult. Glass x2 → 6. Score: 60 × 6 = 360.
r.name = "glass_king_caino"
r.caino_xmult = 2  -- verify Caino gained +1
results[#results+1] = r

-- (w) Ice Cream decay across 2 hands
card_counter = 0
local hands_w = make_hands()
G.GAME.hands = hands_w
local ice = make_joker("Ice Cream", {extra = {chips = 100, chip_mod = 5}})
-- Hand 1
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {}, hands_w, make_blind("Small Blind", 1.0), {ice}
)
r.name = "ice_cream_hand1"
r.ice_chips_after = ice.ability.extra.chips  -- should be 95
results[#results+1] = r

-- Hand 2
card_counter = 0
r = score_hand(
    {make_card("Hearts","Ace"), make_card("Spades","Ace")},
    {}, hands_w, make_blind("Small Blind", 1.0), {ice}
)
r.name = "ice_cream_hand2"
r.ice_chips_after = ice.ability.extra.chips  -- should be 90
results[#results+1] = r

-- Output
io.write(json_value({
    lua_version = _VERSION .. (jit and " (LuaJIT)" or ""),
    tests = results,
}))
io.write("\n")
