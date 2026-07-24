"""Guest entry point for one Episode-local Policy process."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ...._protocol.policy import decode_policy_value, encode_policy_value
from ....policy import Policy, PolicyContext
from .stream import read_policy_message, write_policy_message


def _load_policy(program_directory: Path, context: PolicyContext) -> Policy:
    source = program_directory / "policy.py"
    spec = importlib.util.spec_from_file_location("_evopolicygym_submission", source)
    if spec is None or spec.loader is None:
        raise ImportError("Policy module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return _make_policy(module, context)


def _make_policy(module: ModuleType, context: PolicyContext) -> Policy:
    factory = getattr(module, "make_policy", None)
    if not callable(factory):
        raise TypeError("policy.py must export callable make_policy")
    policy = factory(context)
    if not isinstance(policy, Policy):
        raise TypeError("make_policy() must return an object with act()")
    return policy


def _context_from_message(message: Mapping[str, object]) -> PolicyContext:
    if message.get("type") != "context":
        raise ValueError("first Policy message must contain context")
    policy_seed = message.get("policy_seed")
    if type(policy_seed) is not str:
        raise TypeError("policy_seed is invalid")
    integer_seed = int(policy_seed)
    if str(integer_seed) != policy_seed:
        raise ValueError("policy_seed is not canonical")
    metadata = decode_policy_value(message.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("Policy metadata must be a mapping")
    return PolicyContext(
        observation_space=decode_policy_value(message.get("observation_space")),
        action_space=decode_policy_value(message.get("action_space")),
        metadata=metadata,
        policy_seed=integer_seed,
    )


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        context = _context_from_message(read_policy_message(stdin))
        policy = _load_policy(Path.cwd(), context)
        write_policy_message(stdout, {"type": "ready"})
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        try:
            write_policy_message(stdout, {"type": "error", "code": "exception"})
        except BaseException:
            pass
        return 1

    while True:
        try:
            message = read_policy_message(stdin)
            kind = message.get("type")
            if kind == "close":
                return 0
            if kind != "act":
                raise ValueError("Policy request type is invalid")
            observation = decode_policy_value(message.get("observation"))
            action = policy.act(observation)
            encoded_action = encode_policy_value(action)
            write_policy_message(stdout, {"type": "action", "value": encoded_action})
        except (TypeError, ValueError, RecursionError):
            traceback.print_exc(file=sys.stderr)
            try:
                write_policy_message(
                    stdout,
                    {"type": "error", "code": "protocol_error"},
                )
            except BaseException:
                pass
            return 1
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            try:
                write_policy_message(stdout, {"type": "error", "code": "exception"})
            except BaseException:
                pass
            return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = []
