from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from typing import Any

from evopolicygym import Case, Pool, PoolKind, Task
from evopolicygym.infra.runtime import Roller, Turn


def make_task() -> Task:
    return Task(name="toy", version="0.1", obs={}, act={}, steps=5, cases=8)


@dataclass(slots=True)
class ToyPolicy:
    resets: list[int] = field(default_factory=list)

    def reset(self, episode_index: int) -> None:
        self.resets.append(episode_index)

    def act(self, obs: int) -> int:
        return 1


@dataclass(slots=True)
class ToyWorld:
    state: int = 0
    steps: int = 0

    def reset(self, case: Case) -> int:
        self.state = case.id
        return self.state

    def step(self, action: int) -> Turn:
        self.steps += 1
        self.state += action
        return Turn(
            obs=self.state,
            reward=float(self.state),
            terminated=self.steps == 2,
            info={"steps": self.steps},
        )

    def sample(self) -> int:
        return 0


class BadActPolicy:
    def reset(self, episode_index: int) -> None:
        return None

    def act(self, obs: Any) -> int:
        raise ValueError("bad act")


class BadResetPolicy:
    def reset(self, episode_index: int) -> None:
        raise ValueError("bad reset")

    def act(self, obs: Any) -> int:
        return 1


class SampleWorld(ToyWorld):
    def sample(self) -> int:
        return -1


class TalkPolicy:
    def reset(self, episode_index: int) -> None:
        print(f"reset {episode_index}")
        print("reset err", file=sys.stderr)

    def act(self, obs: Any) -> int:
        print(f"act {obs}")
        print("act err", file=sys.stderr)
        return 1


class RollerTest(unittest.TestCase):
    def test_rolls_cases_into_traces(self) -> None:
        policy = ToyPolicy()
        roller = Roller(ToyWorld)

        traces = roller.run(
            policy,
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            (0, 2),
        )

        self.assertEqual(policy.resets, [0, 1])
        self.assertEqual([trace.episode for trace in traces], [0, 2])
        self.assertEqual([trace.reward for trace in traces], [3.0, 7.0])
        self.assertEqual([len(trace.steps) for trace in traces], [2, 2])
        self.assertEqual(traces[0].steps[0]["obs"], 0)
        self.assertEqual(traces[0].steps[0]["action"], 1)
        self.assertEqual(traces[0].steps[1]["terminated"], True)

    def test_act_error_uses_sample_and_marks_trace_error(self) -> None:
        roller = Roller(SampleWorld)

        trace = roller.run(
            BadActPolicy(),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            (4,),
        )[0]

        self.assertEqual(trace.reward, 3.0)
        self.assertEqual(trace.steps[0]["action"], -1)
        self.assertTrue(trace.error.startswith("act_error:"))

    def test_reset_error_creates_empty_error_trace(self) -> None:
        roller = Roller(ToyWorld)

        trace = roller.run(
            BadResetPolicy(),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            (4,),
        )[0]

        self.assertEqual(trace.reward, 0.0)
        self.assertEqual(trace.steps, ())
        self.assertTrue(trace.error.startswith("reset_error:"))

    def test_captures_policy_stdout_and_stderr(self) -> None:
        roller = Roller(ToyWorld)

        trace = roller.run(
            TalkPolicy(),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            (4,),
        )[0]

        self.assertIn("reset 0", trace.stdout)
        self.assertIn("act 4", trace.stdout)
        self.assertIn("reset err", trace.stderr)
        self.assertIn("act err", trace.stderr)


if __name__ == "__main__":
    unittest.main()
