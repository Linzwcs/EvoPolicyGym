---
locale: en
page: policy
section: core
title: "Policy ABI"
navTitle: "Policy ABI"
description: "The policy/v1 entry point, PolicyContext, PolicyValue carriers, state lifecycle, and failure semantics."
lead: "One fixed factory, one action method, and a bounded value surface visible to submitted code."
index: D3
order: 3
docsVersion: v0.3
status: draft
---

## Program entry point

Every Program directory must contain `policy.py`. The fixed entry point is
`policy.py:make_policy`:

```python
from evopolicygym.policy import PolicyContext, PolicyValue


class MyPolicy:
    def __init__(self, context: PolicyContext):
        self._seed = context.policy_seed

    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0


def make_policy(context: PolicyContext) -> MyPolicy:
    return MyPolicy(context)
```

`make_policy(context)` runs once for each Episode. The returned object must
provide `act(observation)`.

There is no Policy-visible `learn()`, `reset()`, `update()`, Submission, or
Feedback method.

## PolicyContext

| Field | Meaning |
| --- | --- |
| `observation_space` | Public description of Policy-visible observations. |
| `action_space` | Public description of admissible Actions. |
| `metadata` | String-keyed, Case-independent Benchmark metadata. |
| `policy_seed` | An unsigned 64-bit seed for this fresh Policy instance. |

`PolicyContext` never contains Case identity, Environment seed, Host paths,
file descriptors, credentials, pool identity, scores, runtime evidence, or
scorer objects.

## PolicyValue

Values crossing the Policy boundary must be composed only from:

```text
None | bool | int | float | str | bytes
| TensorValue
| list[PolicyValue]
| tuple[PolicyValue, ...]
| dict[str, PolicyValue]
```

The rules are strict:

- floats must be finite;
- integers must fit signed or unsigned 64-bit;
- mapping keys must be exact strings;
- containers are validated and detached recursively;
- custom Python objects and pickle graphs are rejected.

## TensorValue

`TensorValue` carries a canonical dense tensor:

```python
from evopolicygym.policy import TensorValue

pixels = TensorValue(
    dtype="uint8",
    shape=(84, 84, 3),
    data=raw_rgb_bytes,
)
```

Supported dtypes are `bool`, unsigned and signed integers from 8 to 64 bits,
and `float16`, `float32`, and `float64`. Multibyte values are little-endian,
floating values must be finite, and the byte length must exactly match the
shape and dtype.

## Episode-local state

```text
Episode 0  new process → make_policy() → act() × N → destroy
Episode 1  new process → make_policy() → act() × N → destroy
Episode N  new process → make_policy() → act() × N → destroy
```

A Policy may retain history, recurrent state, search trees, caches, or
temporary parameters between `act()` calls in the same Episode. No state
crosses into another Episode.

Cross-Episode learning belongs to the outer Coding Agent, which authors and
submits a new immutable Program.

## Action semantics

The Environment receives the complete, unmodified Action returned by
`Policy.act()`.

```text
Policy.act(observation)
        ↓
PolicyValue validation
        ↓
Environment.step(action)
        ├── valid   → trusted Step
        └── invalid → InvalidAction
                       Policy failure; no fallback step
```

Invalid Actions are never clipped, repaired, sampled, or replaced.

## Policy failures

The public Policy failure codes are:

| Code | Meaning |
| --- | --- |
| `exception` | `make_policy()` or `act()` raised. |
| `timeout` | The Policy exceeded its Episode operation timeout. |
| `invalid_action` | The Environment rejected the complete Action. |
| `protocol_error` | The Policy process returned a malformed or unencodable value. |

After Policy failure, the evaluator does not call `Environment.step()` again.
Trusted Environment and execution faults remain separate and abort Evaluation.

## Next

- [Evaluation and Runs →](../evaluation/)
- [Execution and safety →](../runtime/)
- [Read `policy.py` source ↗](https://github.com/Linzwcs/EvoPolicyGym/blob/main/src/evopolicygym/policy.py)
