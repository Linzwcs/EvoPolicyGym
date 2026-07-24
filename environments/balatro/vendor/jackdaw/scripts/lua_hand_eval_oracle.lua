#!/usr/bin/env luajit
--- Hand evaluation oracle: runs the source's evaluate_poker_hand and outputs
--- results as JSON for cross-validation against the Python port.
---
--- Usage:
---   luajit scripts/lua_hand_eval_oracle.lua
---
--- Compatible with LuaJIT 2.1.

-- Resolve project root from arg[0]
local script_path = arg[0] or ""
local project_root = ""
-- Try to find "scripts/" in the path and go up one level
local scripts_pos = script_path:find("scripts")
if scripts_pos then
    project_root = script_path:sub(1, scripts_pos - 1)
end
if project_root == "" then project_root = "./" end

----------------------------------------------------------------------------
-- Stub globals needed by find_joker
----------------------------------------------------------------------------

G = {
    jokers = { cards = {} },       -- no jokers by default
    consumeables = { cards = {} },
}

function find_joker(name, non_debuff)
    local jokers = {}
    if not G.jokers or not G.jokers.cards then return {} end
    for k, v in pairs(G.jokers.cards) do
        if v and type(v) == 'table' and v.ability and v.ability.name == name
           and (non_debuff or not v.debuff) then
            table.insert(jokers, v)
        end
    end
    for k, v in pairs(G.consumeables.cards) do
        if v and type(v) == 'table' and v.ability and v.ability.name == name
           and (non_debuff or not v.debuff) then
            table.insert(jokers, v)
        end
    end
    return jokers
end

----------------------------------------------------------------------------
-- Load source hand eval functions
----------------------------------------------------------------------------

local source_path = project_root .. "balatro_source/functions/misc_functions.lua"
local f = assert(io.open(source_path, "r"))
local src = f:read("*a")
f:close()

-- Extract evaluate_poker_hand, get_flush, get_straight, get_X_same, get_highest
local func_src = ""
local n = 0
for line in src:gmatch("[^\n]*\n?") do
    n = n + 1
    if n >= 376 and n <= 621 then
        func_src = func_src .. line
    end
end

-- Load the functions into the global environment
local chunk, err = loadstring(func_src)
if not chunk then error("Failed to parse hand eval functions: " .. (err or "?")) end
chunk()

----------------------------------------------------------------------------
-- Card constructor helper
----------------------------------------------------------------------------

local function make_card(suit, value, opts)
    opts = opts or {}
    local id_map = {
        ["2"]=2, ["3"]=3, ["4"]=4, ["5"]=5, ["6"]=6, ["7"]=7,
        ["8"]=8, ["9"]=9, ["10"]=10, ["Jack"]=11, ["Queen"]=12,
        ["King"]=13, ["Ace"]=14,
    }
    local nominal_map = {
        ["2"]=2, ["3"]=3, ["4"]=4, ["5"]=5, ["6"]=6, ["7"]=7,
        ["8"]=8, ["9"]=9, ["10"]=10, ["Jack"]=10, ["Queen"]=10,
        ["King"]=10, ["Ace"]=11,
    }
    local suit_nom = ({Spades=0.04, Hearts=0.03, Clubs=0.02, Diamonds=0.01})[suit]
    local face_nom = ({Jack=0.1, Queen=0.2, King=0.3, Ace=0.4})[value] or 0

    local card = {
        base = {
            suit = suit,
            value = value,
            id = id_map[value],
            nominal = nominal_map[value],
            suit_nominal = suit_nom,
            face_nominal = face_nom,
        },
        ability = {
            name = opts.name or "Default Base",
            effect = opts.effect or "",
        },
        debuff = opts.debuff or false,
        sort_id = opts.sort_id or 0,
        unique_val = opts.unique_val or math.random(),
    }

    -- Methods matching card.lua
    function card:get_id()
        if self.ability.effect == "Stone Card" then return -math.random(100, 1000000) end
        return self.base.id
    end

    function card:get_nominal(mod)
        local mult = 1
        if mod == 'suit' then mult = 1000 end
        if self.ability.effect == 'Stone Card' then mult = -1000 end
        return self.base.nominal + self.base.suit_nominal*mult + self.base.face_nominal + 0.000001*self.unique_val
    end

    function card:is_suit(suit, bypass_debuff, flush_calc)
        if flush_calc then
            if self.ability.effect == 'Stone Card' then return false end
            if self.ability.name == "Wild Card" and not self.debuff then return true end
            -- No smeared joker in oracle (no G.jokers mock for it)
            return self.base.suit == suit
        else
            if self.debuff and not bypass_debuff then return end
            if self.ability.effect == 'Stone Card' then return false end
            if self.ability.name == "Wild Card" then return true end
            return self.base.suit == suit
        end
    end

    return card
end

----------------------------------------------------------------------------
-- JSON helper
----------------------------------------------------------------------------

local function json_string(s) return '"' .. s:gsub('\\', '\\\\'):gsub('"', '\\"') .. '"' end

local function json_value(v)
    if v == nil then return "null"
    elseif type(v) == "boolean" then return v and "true" or "false"
    elseif type(v) == "number" then return tostring(v)
    elseif type(v) == "string" then return json_string(v)
    elseif type(v) == "table" then
        if #v > 0 then
            local parts = {}
            for i = 1, #v do parts[i] = json_value(v[i]) end
            return "[" .. table.concat(parts, ", ") .. "]"
        elseif next(v) == nil then
            return "[]"
        else
            local parts = {}
            local keys = {}
            for k in pairs(v) do keys[#keys+1] = tostring(k) end
            table.sort(keys)
            for _, k in ipairs(keys) do
                parts[#parts+1] = json_string(k) .. ": " .. json_value(v[k])
            end
            return "{" .. table.concat(parts, ", ") .. "}"
        end
    end
    return "null"
end

----------------------------------------------------------------------------
-- Run a test case
----------------------------------------------------------------------------

local function card_desc(c)
    return c.base.value .. " of " .. c.base.suit
end

local function run_test(name, hand, joker_names)
    -- Set up joker stubs
    G.jokers.cards = {}
    if joker_names then
        for _, jname in ipairs(joker_names) do
            G.jokers.cards[#G.jokers.cards+1] = {ability = {name = jname}}
        end
    end

    local results = evaluate_poker_hand(hand)

    -- Find detected hand (priority order)
    local priority = {
        "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
        "Four of a Kind", "Full House", "Flush", "Straight",
        "Three of a Kind", "Two Pair", "Pair", "High Card",
    }
    local detected = "NULL"
    for _, h in ipairs(priority) do
        if next(results[h] or {}) then
            detected = h
            break
        end
    end

    -- Collect which hands are populated
    local populated = {}
    for _, h in ipairs(priority) do
        if next(results[h] or {}) then
            populated[#populated+1] = h
        end
    end

    -- Card descriptions for scoring hand
    local scoring_descs = {}
    if results[detected] and results[detected][1] then
        for _, c in ipairs(results[detected][1]) do
            scoring_descs[#scoring_descs+1] = card_desc(c)
        end
    end

    return {
        name = name,
        detected = detected,
        populated = populated,
        scoring_cards = scoring_descs,
        card_count = #hand,
    }
end

----------------------------------------------------------------------------
-- Test cases
----------------------------------------------------------------------------

local test_results = {}

-- High Card
test_results[#test_results+1] = run_test("high_card", {
    make_card("Hearts", "2"), make_card("Spades", "5"),
    make_card("Clubs", "8"), make_card("Diamonds", "Jack"),
    make_card("Hearts", "Ace"),
})

-- Pair
test_results[#test_results+1] = run_test("pair", {
    make_card("Hearts", "5"), make_card("Spades", "5"),
    make_card("Clubs", "8"), make_card("Diamonds", "Jack"),
    make_card("Hearts", "Ace"),
})

-- Two Pair
test_results[#test_results+1] = run_test("two_pair", {
    make_card("Hearts", "5"), make_card("Spades", "5"),
    make_card("Clubs", "Jack"), make_card("Diamonds", "Jack"),
    make_card("Hearts", "Ace"),
})

-- Three of a Kind
test_results[#test_results+1] = run_test("three_of_a_kind", {
    make_card("Hearts", "King"), make_card("Spades", "King"),
    make_card("Clubs", "King"), make_card("Diamonds", "5"),
    make_card("Hearts", "2"),
})

-- Straight
test_results[#test_results+1] = run_test("straight", {
    make_card("Hearts", "4"), make_card("Spades", "5"),
    make_card("Clubs", "6"), make_card("Diamonds", "7"),
    make_card("Hearts", "8"),
})

-- Ace-low straight
test_results[#test_results+1] = run_test("ace_low_straight", {
    make_card("Hearts", "Ace"), make_card("Spades", "2"),
    make_card("Clubs", "3"), make_card("Diamonds", "4"),
    make_card("Hearts", "5"),
})

-- Flush
test_results[#test_results+1] = run_test("flush", {
    make_card("Hearts", "2"), make_card("Hearts", "5"),
    make_card("Hearts", "8"), make_card("Hearts", "Jack"),
    make_card("Hearts", "Ace"),
})

-- Full House
test_results[#test_results+1] = run_test("full_house", {
    make_card("Hearts", "King"), make_card("Spades", "King"),
    make_card("Clubs", "King"), make_card("Diamonds", "5"),
    make_card("Hearts", "5"),
})

-- Four of a Kind
test_results[#test_results+1] = run_test("four_of_a_kind", {
    make_card("Hearts", "7"), make_card("Spades", "7"),
    make_card("Clubs", "7"), make_card("Diamonds", "7"),
    make_card("Hearts", "Ace"),
})

-- Straight Flush
test_results[#test_results+1] = run_test("straight_flush", {
    make_card("Hearts", "4"), make_card("Hearts", "5"),
    make_card("Hearts", "6"), make_card("Hearts", "7"),
    make_card("Hearts", "8"),
})

-- Five of a Kind
test_results[#test_results+1] = run_test("five_of_a_kind", {
    make_card("Hearts", "Ace"), make_card("Spades", "Ace"),
    make_card("Clubs", "Ace"), make_card("Diamonds", "Ace"),
    make_card("Hearts", "Ace"),
})

-- Flush Five
test_results[#test_results+1] = run_test("flush_five", {
    make_card("Hearts", "Ace"), make_card("Hearts", "Ace"),
    make_card("Hearts", "Ace"), make_card("Hearts", "Ace"),
    make_card("Hearts", "Ace"),
})

-- Flush House
test_results[#test_results+1] = run_test("flush_house", {
    make_card("Hearts", "King"), make_card("Hearts", "King"),
    make_card("Hearts", "King"), make_card("Hearts", "5"),
    make_card("Hearts", "5"),
})

-- Four Fingers: 4-card flush
test_results[#test_results+1] = run_test("four_fingers_flush", {
    make_card("Clubs", "3"), make_card("Clubs", "7"),
    make_card("Clubs", "10"), make_card("Clubs", "King"),
    make_card("Hearts", "Ace"),
}, {"Four Fingers"})

-- Shortcut straight
test_results[#test_results+1] = run_test("shortcut_straight", {
    make_card("Hearts", "3"), make_card("Spades", "5"),
    make_card("Clubs", "6"), make_card("Diamonds", "7"),
    make_card("Hearts", "8"),
}, {"Shortcut"})

-- No wrap (Q-K-A-2-3)
test_results[#test_results+1] = run_test("no_wrap", {
    make_card("Hearts", "Queen"), make_card("Spades", "King"),
    make_card("Clubs", "Ace"), make_card("Diamonds", "2"),
    make_card("Hearts", "3"),
})

-- Wild Card in flush
test_results[#test_results+1] = run_test("wild_in_flush", {
    make_card("Spades", "2"), make_card("Spades", "5"),
    make_card("Spades", "8"), make_card("Spades", "Jack"),
    make_card("Hearts", "Ace", {name = "Wild Card", effect = "Wild Card"}),
})

-- Output
io.write(json_value({
    lua_version = _VERSION .. (jit and " (LuaJIT)" or ""),
    tests = test_results,
}))
io.write("\n")
