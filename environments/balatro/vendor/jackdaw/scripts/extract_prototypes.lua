#!/usr/bin/env luajit
--- Extract prototype data tables from Balatro's game.lua into JSON files.
---
--- Usage:
---   luajit scripts/extract_prototypes.lua
---
--- Outputs:
---   jackdaw/engine/data/centers.json
---   jackdaw/engine/data/cards.json
---   jackdaw/engine/data/blinds.json
---   jackdaw/engine/data/tags.json
---   jackdaw/engine/data/stakes.json
---   jackdaw/engine/data/seals.json
---
--- Compatible with LuaJIT 2.1.

-- Resolve project root from script location (works on both Windows and Unix)
local script_dir = arg[0]:match("(.*[/\\])") or "./"
local project_root = script_dir:gsub("[/\\]scripts[/\\]$", "/")
-- Fallback: if script_dir doesn't contain 'scripts', try relative
if not script_dir:find("scripts") then
    project_root = "./"
end
local source_path = project_root .. "balatro_source/game.lua"

----------------------------------------------------------------------------
-- JSON serializer (handles nested tables, booleans, nil → null)
----------------------------------------------------------------------------

local function json_string(s)
    s = s:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
    return '"' .. s .. '"'
end

local json_value  -- forward declaration

local function is_array(t)
    local max_idx = 0
    local count = 0
    for k, _ in pairs(t) do
        if type(k) ~= "number" or k ~= math.floor(k) or k < 1 then
            return false
        end
        if k > max_idx then max_idx = k end
        count = count + 1
    end
    return count == max_idx
end

json_value = function(v, indent, level)
    indent = indent or 2
    level = level or 0
    local t = type(v)
    if v == nil then
        return "null"
    elseif t == "boolean" then
        return v and "true" or "false"
    elseif t == "number" then
        if v == math.floor(v) and math.abs(v) < 2^53 then
            return string.format("%d", v)
        else
            return string.format("%.15g", v)
        end
    elseif t == "string" then
        return json_string(v)
    elseif t == "table" then
        if is_array(v) then
            if #v == 0 then return "[]" end
            local parts = {}
            local pad = string.rep(" ", indent * (level + 1))
            for i = 1, #v do
                parts[i] = pad .. json_value(v[i], indent, level + 1)
            end
            return "[\n" .. table.concat(parts, ",\n") .. "\n" .. string.rep(" ", indent * level) .. "]"
        else
            local keys = {}
            for k, _ in pairs(v) do keys[#keys + 1] = k end
            table.sort(keys, function(a, b)
                -- Sort: "key" first, then "name", then "set", then alphabetical
                local priority = {key=0, name=1, set=2, order=3, rarity=4, cost=5}
                local pa = priority[tostring(a)] or 99
                local pb = priority[tostring(b)] or 99
                if pa ~= pb then return pa < pb end
                return tostring(a) < tostring(b)
            end)
            if #keys == 0 then return "{}" end
            local parts = {}
            local pad = string.rep(" ", indent * (level + 1))
            for _, k in ipairs(keys) do
                parts[#parts + 1] = pad .. json_string(tostring(k)) .. ": " .. json_value(v[k], indent, level + 1)
            end
            return "{\n" .. table.concat(parts, ",\n") .. "\n" .. string.rep(" ", indent * level) .. "}"
        end
    else
        return '"<' .. t .. '>"'
    end
end

local function write_json(path, data)
    local f = assert(io.open(path, "w"))
    f:write(json_value(data))
    f:write("\n")
    f:close()
    io.write("  Written: " .. path .. "\n")
end

----------------------------------------------------------------------------
-- Mock environment for loading game.lua's init_item_prototypes
----------------------------------------------------------------------------

-- Minimal stubs for functions used in init_item_prototypes
local function HEX(hex)
    -- Convert hex color to RGBA table (matching the source's HEX function)
    local r = tonumber(hex:sub(1,2), 16) / 255
    local g = tonumber(hex:sub(3,4), 16) / 255
    local b = tonumber(hex:sub(5,6), 16) / 255
    return {r, g, b, 1}
end

local function localize(...)
    return "localized"
end

-- Create the Game mock
local Game = {}
Game.__index = Game

function Game:new()
    local o = setmetatable({}, self)
    return o
end

----------------------------------------------------------------------------
-- Load and execute init_item_prototypes
----------------------------------------------------------------------------

io.write("Loading " .. source_path .. "...\n")
local f = assert(io.open(source_path, "r"))
local source = f:read("*a")
f:close()

-- Extract just the init_item_prototypes function body
local func_start = source:find("function Game:init_item_prototypes%(%)")
local func_end = source:find("\nfunction Game:", func_start + 1)
if not func_start or not func_end then
    error("Could not find init_item_prototypes in " .. source_path)
end

local func_body = source:sub(func_start, func_end - 1)

-- Remove function header
func_body = func_body:gsub("function Game:init_item_prototypes%(%)", "")
-- Remove the trailing 'end'
func_body = func_body:gsub("%s*end%s*$", "")

-- Cut off after P_JOKER_RARITY_POOLS / P_LOCKED (before save_progress and love.filesystem)
-- We only need the data definitions, not the pool-building or unlock logic
local cutoff = func_body:find("self:save_progress")
if cutoff then
    func_body = func_body:sub(1, cutoff - 1)
end

-- Create the game object
local game_obj = {}

-- Build the execution environment
local env = setmetatable({
    self = game_obj,
    HEX = HEX,
    localize = localize,
    string = string,
    table = table,
    math = math,
    pairs = pairs,
    ipairs = ipairs,
    type = type,
    tostring = tostring,
    tonumber = tonumber,
    print = print,
    next = next,
    unpack = unpack,
}, {__index = _G})

-- Compile and execute
local chunk, err = loadstring(func_body)
if not chunk then
    error("Failed to parse init_item_prototypes: " .. (err or "unknown error"))
end
setfenv(chunk, env)

io.write("Executing init_item_prototypes...\n")
chunk()

----------------------------------------------------------------------------
-- Post-process: add 'key' field to each entry
----------------------------------------------------------------------------

local function add_keys(tbl)
    for k, v in pairs(tbl) do
        if type(v) == "table" then
            v.key = k
        end
    end
end

add_keys(game_obj.P_CENTERS or {})
add_keys(game_obj.P_CARDS or {})
add_keys(game_obj.P_BLINDS or {})
add_keys(game_obj.P_TAGS or {})
add_keys(game_obj.P_STAKES or {})
add_keys(game_obj.P_SEALS or {})

----------------------------------------------------------------------------
-- Write JSON files
----------------------------------------------------------------------------

local out_dir = project_root .. "jackdaw/engine/data/"

io.write("\nWriting JSON files...\n")
write_json(out_dir .. "centers.json", game_obj.P_CENTERS or {})
write_json(out_dir .. "cards.json", game_obj.P_CARDS or {})
write_json(out_dir .. "blinds.json", game_obj.P_BLINDS or {})
write_json(out_dir .. "tags.json", game_obj.P_TAGS or {})
write_json(out_dir .. "stakes.json", game_obj.P_STAKES or {})
write_json(out_dir .. "seals.json", game_obj.P_SEALS or {})

----------------------------------------------------------------------------
-- Print summary counts
----------------------------------------------------------------------------

io.write("\n=== Prototype Counts ===\n")

-- Count P_CENTERS by set
local set_counts = {}
local total_centers = 0
for k, v in pairs(game_obj.P_CENTERS or {}) do
    total_centers = total_centers + 1
    local s = v.set or "Unknown"
    set_counts[s] = (set_counts[s] or 0) + 1
end

local sorted_sets = {}
for s, _ in pairs(set_counts) do sorted_sets[#sorted_sets + 1] = s end
table.sort(sorted_sets)

for _, s in ipairs(sorted_sets) do
    io.write(string.format("  P_CENTERS[%s]: %d\n", s, set_counts[s]))
end
io.write(string.format("  P_CENTERS total: %d\n", total_centers))

-- Count other tables
local function count_table(t)
    local n = 0
    for _ in pairs(t or {}) do n = n + 1 end
    return n
end

io.write(string.format("  P_CARDS: %d\n", count_table(game_obj.P_CARDS)))
io.write(string.format("  P_BLINDS: %d\n", count_table(game_obj.P_BLINDS)))
io.write(string.format("  P_TAGS: %d\n", count_table(game_obj.P_TAGS)))
io.write(string.format("  P_STAKES: %d\n", count_table(game_obj.P_STAKES)))
io.write(string.format("  P_SEALS: %d\n", count_table(game_obj.P_SEALS)))

-- Count jokers by rarity
local rarity_counts = {0, 0, 0, 0}
for k, v in pairs(game_obj.P_CENTERS or {}) do
    if v.set == "Joker" and v.rarity then
        rarity_counts[v.rarity] = rarity_counts[v.rarity] + 1
    end
end
io.write("\n=== Joker Rarity Breakdown ===\n")
io.write(string.format("  Common (1): %d\n", rarity_counts[1]))
io.write(string.format("  Uncommon (2): %d\n", rarity_counts[2]))
io.write(string.format("  Rare (3): %d\n", rarity_counts[3]))
io.write(string.format("  Legendary (4): %d\n", rarity_counts[4]))
