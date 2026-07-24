#!/usr/bin/env lua
--- RNG oracle: generates ground-truth PRNG outputs for cross-validation.
---
--- Usage:
---   lua scripts/lua_rng_oracle.lua [SEED ...]
---   luajit scripts/lua_rng_oracle.lua TESTSEED A3K9NZ2B TUTORIAL
---
--- If no seeds given, uses a built-in set.  Outputs JSON to stdout.
---
--- Compatible with Lua 5.1, 5.2, 5.3, 5.4, and LuaJIT.

----------------------------------------------------------------------------
-- Exact copies from balatro_source/functions/misc_functions.lua
-- (no LÖVE2D or G.GAME dependencies)
----------------------------------------------------------------------------

function pseudohash(str)
  if true then
    local num = 1
    for i = #str, 1, -1 do
      num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
    end
    return num
  end
end

function pseudoseed_stateful(key, state)
  -- Mirrors the normal (non-predict) path of pseudoseed()
  -- but uses an explicit state table instead of G.GAME.pseudorandom
  if not state[key] then
    state[key] = pseudohash(key .. (state.seed or ""))
  end
  state[key] = math.abs(tonumber(string.format("%.13f",
    (2.134453429141 + state[key] * 1.72431234) % 1)))
  return (state[key] + (state.hashed_seed or 0)) / 2
end

function pseudoseed_predict(key, predict_seed)
  local _pseed = pseudohash(key .. (predict_seed or ""))
  _pseed = math.abs(tonumber(string.format("%.13f",
    (2.134453429141 + _pseed * 1.72431234) % 1)))
  return (_pseed + (pseudohash(predict_seed) or 0)) / 2
end

function pseudorandom_from_seed(seed, min, max)
  -- The real game does: math.randomseed(seed); math.random(...)
  -- This works in LuaJIT/Lua 5.1 with float seeds.
  -- In Lua 5.3+ math.randomseed needs an integer.
  -- We handle both cases.
  local ok, _ = pcall(math.randomseed, seed)
  if not ok then
    -- Lua 5.3/5.4: convert float seed to integer the same way the engine
    -- effectively does (the float is truncated to an integer seed).
    math.randomseed(math.floor(seed * 2^53))
  end
  if min and max then
    return math.random(min, max)
  else
    return math.random()
  end
end

function pseudoshuffle(list, seed)
  local ok, _ = pcall(math.randomseed, seed)
  if not ok then
    math.randomseed(math.floor(seed * 2^53))
  end
  for i = #list, 2, -1 do
    local j = math.random(i)
    list[i], list[j] = list[j], list[i]
  end
end

function pseudorandom_element(t, seed)
  local ok, _ = pcall(math.randomseed, seed)
  if not ok then
    math.randomseed(math.floor(seed * 2^53))
  end
  local keys = {}
  for k, v in pairs(t) do
    keys[#keys + 1] = { k = k, v = v }
  end
  table.sort(keys, function(a, b) return a.k < b.k end)
  local key = keys[math.random(#keys)].k
  return t[key], key
end

----------------------------------------------------------------------------
-- JSON helper (minimal, handles our output types)
----------------------------------------------------------------------------

local function json_string(s)
  return '"' .. s:gsub('\\', '\\\\'):gsub('"', '\\"') .. '"'
end

local function json_value(v)
  local t = type(v)
  if t == "number" then
    return string.format("%.15g", v)
  elseif t == "string" then
    return json_string(v)
  elseif t == "boolean" then
    return v and "true" or "false"
  elseif t == "table" then
    -- detect array vs object
    if #v > 0 or next(v) == nil then
      -- array
      local parts = {}
      for i = 1, #v do parts[i] = json_value(v[i]) end
      return "[" .. table.concat(parts, ", ") .. "]"
    else
      -- object
      local parts = {}
      -- sort keys for deterministic output
      local sorted_keys = {}
      for k, _ in pairs(v) do sorted_keys[#sorted_keys + 1] = k end
      table.sort(sorted_keys)
      for _, k in ipairs(sorted_keys) do
        parts[#parts + 1] = json_string(k) .. ": " .. json_value(v[k])
      end
      return "{" .. table.concat(parts, ", ") .. "}"
    end
  else
    return "null"
  end
end

----------------------------------------------------------------------------
-- Test configuration
----------------------------------------------------------------------------

local STREAM_KEYS = {
  "boss", "shuffle", "lucky_mult", "rarity1", "stdset1",
  "cdt1", "front1", "edition_generic",
}

local PREDICT_PAIRS = {
  { "Joker4", nil },  -- predict_seed uses the test seed
  { "boss",   nil },
  { "soul_Joker1", nil },
}

local N_SEED_ADVANCES  = 10
local N_RANDOM_CALLS   = 5

local DEFAULT_SEEDS = { "TESTSEED", "A3K9NZ2B", "TUTORIAL" }

----------------------------------------------------------------------------
-- Generate oracle data for one seed
----------------------------------------------------------------------------

local function generate_for_seed(seed_str)
  local result = {}
  result.seed = seed_str
  result.hashed_seed = pseudohash(seed_str)

  -- pseudohash of stream key concatenations
  result.pseudohash = {}
  for _, key in ipairs(STREAM_KEYS) do
    result.pseudohash[key] = pseudohash(key .. seed_str)
  end

  -- pseudoseed: N consecutive advances per stream
  result.pseudoseed = {}
  local state = { seed = seed_str, hashed_seed = pseudohash(seed_str) }

  for _, key in ipairs(STREAM_KEYS) do
    local stream_data = {}
    for call = 1, N_SEED_ADVANCES do
      local ret = pseudoseed_stateful(key, state)
      stream_data[call] = {
        call    = call,
        result  = ret,
        stored  = state[key],
      }
    end
    result.pseudoseed[key] = stream_data
  end

  -- pseudorandom: integer range [1,10] from fresh state per stream
  result.pseudorandom_int = {}
  local state2 = { seed = seed_str, hashed_seed = pseudohash(seed_str) }

  for _, key in ipairs(STREAM_KEYS) do
    local calls = {}
    for call = 1, N_RANDOM_CALLS do
      local sv = pseudoseed_stateful(key, state2)
      local r  = pseudorandom_from_seed(sv, 1, 10)
      calls[call] = {
        call      = call,
        seed_val  = sv,
        int_result = r,
      }
    end
    result.pseudorandom_int[key] = calls
  end

  -- pseudorandom: float output from fresh state per stream
  result.pseudorandom_float = {}
  local state3 = { seed = seed_str, hashed_seed = pseudohash(seed_str) }

  for _, key in ipairs(STREAM_KEYS) do
    local calls = {}
    for call = 1, N_RANDOM_CALLS do
      local sv = pseudoseed_stateful(key, state3)
      local r  = pseudorandom_from_seed(sv, nil, nil)
      calls[call] = {
        call         = call,
        seed_val     = sv,
        float_result = r,
      }
    end
    result.pseudorandom_float[key] = calls
  end

  -- predict_seed (stateless)
  result.predict_seed = {}
  for _, pair in ipairs(PREDICT_PAIRS) do
    local key       = pair[1]
    local pred_seed = pair[2] or seed_str
    local r = pseudoseed_predict(key, pred_seed)
    result.predict_seed[#result.predict_seed + 1] = {
      key         = key,
      predict_with = pred_seed,
      result       = r,
    }
  end

  -- pseudoshuffle: shuffle [1..10] using seed from pseudoseed('shuffle')
  result.shuffle = {}
  local state4 = { seed = seed_str, hashed_seed = pseudohash(seed_str) }

  for trial = 1, 3 do
    local sv  = pseudoseed_stateful("shuffle", state4)
    local lst = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
    pseudoshuffle(lst, sv)
    result.shuffle[trial] = {
      trial     = trial,
      seed_val  = sv,
      result    = lst,
    }
  end

  -- pseudorandom_element: select from a small dict
  result.element = {}
  local test_table = { alpha = 1, beta = 2, gamma = 3, delta = 4, epsilon = 5 }
  local state5 = { seed = seed_str, hashed_seed = pseudohash(seed_str) }

  for trial = 1, 5 do
    local sv = pseudoseed_stateful("element_test", state5)
    local val, key = pseudorandom_element(test_table, sv)
    result.element[trial] = {
      trial    = trial,
      seed_val = sv,
      key      = key,
      value    = val,
    }
  end

  return result
end

----------------------------------------------------------------------------
-- Main
----------------------------------------------------------------------------

local seeds = {}
if #arg > 0 then
  for _, s in ipairs(arg) do
    seeds[#seeds + 1] = s
  end
else
  seeds = DEFAULT_SEEDS
end

-- Detect Lua version for metadata
local lua_version = _VERSION or "unknown"
local is_jit = (jit ~= nil)

local all_results = {}
for _, seed_str in ipairs(seeds) do
  all_results[#all_results + 1] = generate_for_seed(seed_str)
end

-- Output wrapper
local output = {
  lua_version = lua_version .. (is_jit and " (LuaJIT)" or ""),
  note = "Ground-truth RNG oracle output. pseudoseed/pseudohash values are "
      .. "Lua-version-independent. pseudorandom/shuffle/element values depend "
      .. "on math.random implementation and may differ between Lua versions.",
  seeds = all_results,
}

io.write(json_value(output))
io.write("\n")
