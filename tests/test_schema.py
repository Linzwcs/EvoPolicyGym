from __future__ import annotations

import unittest

from evopolicygym import (
    Budget,
    Eval,
    Feed,
    Pick,
    PoolKind,
    Run,
    Score,
    SubmitRecord,
    Verdict,
)
from evopolicygym.protocol import outcome as build_outcome
from evopolicygym.protocol import record, summary


def make_run(*, used: int = 0) -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=4, used=used),
    )


class SummarySchemaTest(unittest.TestCase):
    def test_summary_maps_ok_feed_to_protocol_shape(self) -> None:
        run = make_run(used=2)
        submit = SubmitRecord(index=2, cases=(0, 1))
        feed = Feed(
            submit=2,
            verdict=Verdict.ok,
            cost=2,
            score=Score(mean=2.0, std=1.0, returns=(1.0, 3.0)),
        )

        data = summary(
            run,
            submit,
            feed,
            started="2026-01-01T00:00:00Z",
            completed="2026-01-01T00:00:02Z",
            wall=2.0,
            first=10,
            lengths=(5, 7),
            timeouts=(1,),
            errors=(),
        )

        self.assertEqual(data["schema_version"], "0.1")
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["env_instances"], [0, 1])
        self.assertEqual(data["remaining_budget"], 2)
        self.assertEqual(data["returns"], [1.0, 3.0])
        self.assertEqual(data["min_return"], 1.0)
        self.assertEqual(data["max_return"], 3.0)
        self.assertEqual(data["mean_episode_length"], 6.0)
        self.assertEqual(data["timeouts"], [1])

    def test_summary_rejects_ok_feed_without_returns(self) -> None:
        with self.assertRaisesRegex(ValueError, "submit returns"):
            summary(
                make_run(used=2),
                SubmitRecord(index=1, cases=(0, 1)),
                Feed(
                    submit=1,
                    verdict=Verdict.ok,
                    cost=2,
                    score=Score(mean=2.0, std=1.0),
                ),
                started="2026-01-01T00:00:00Z",
                completed="2026-01-01T00:00:02Z",
                wall=2.0,
                first=0,
                lengths=(5, 7),
            )

    def test_summary_uses_zero_episodes_for_phase_one_reject(self) -> None:
        data = summary(
            make_run(),
            SubmitRecord(index=1, cases=(0, 1, 2)),
            Feed(
                submit=1,
                verdict=Verdict.budget_invalid,
                cost=3,
                score=Score(mean=None, std=None),
            ),
            started="2026-01-01T00:00:00Z",
            completed="2026-01-01T00:00:00Z",
            wall=0.0,
        )

        self.assertEqual(data["n_episodes"], 0)
        self.assertIsNone(data["returns"])
        self.assertEqual(data["remaining_budget"], 4)


class OutcomeSchemaTest(unittest.TestCase):
    def test_outcome_maps_completed_run(self) -> None:
        pick = Pick.from_vals(
            {
                1: Score(mean=3.0, std=0.1),
                2: Score(mean=4.0, std=0.2),
            }
        )
        run = make_run(used=4).done(pick)
        final = Eval(
            kind=PoolKind.final,
            snap=2,
            pool="heldout",
            score=Score(mean=8.0, std=0.5, value=90.0, returns=(7.0, 9.0)),
        )

        data = build_outcome(run, final, auxiliary={"episodes_used": 4})

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["final_score"], 90.0)
        self.assertEqual(data["best_submit_index"], 2)
        self.assertEqual(data["val_scores"], {"1": 3.0, "2": 4.0})
        self.assertEqual(data["heldout_returns"], [7.0, 9.0])
        self.assertEqual(data["auxiliary"], {"episodes_used": 4})

    def test_outcome_maps_no_ok_submit(self) -> None:
        run = make_run().done(Pick.from_vals({}))

        data = build_outcome(run, None)

        self.assertEqual(data["status"], "no_ok_submit")
        self.assertEqual(data["final_score"], 0.0)
        self.assertIsNone(data["best_submit_index"])
        self.assertIsNone(data["val_scores"])

    def test_record_builds_top_level_run_json(self) -> None:
        run = make_run()
        result = {"status": "no_ok_submit"}

        data = record(
            run,
            result,
            dimensions={"episode_budget": 4},
            timing={"wall_time_seconds": 1.0},
            versions={"harness": "0.1.0"},
        )

        self.assertEqual(data["schema_version"], "0.1")
        self.assertEqual(data["protocol_version"], "protocol/v2.0-draft")
        self.assertEqual(data["model"], "agent")
        self.assertEqual(data["outcome"], result)
        self.assertEqual(data["artifacts"]["workspace"], "workspace/")
        self.assertEqual(data["artifacts"]["feedback"], "workspace/feedback/")
        self.assertEqual(data["versions"], {"harness": "0.1.0"})

    def test_outcome_requires_error_details_for_error_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "error details"):
            build_outcome(make_run().fail(), None)

        data = build_outcome(
            make_run().fail(),
            None,
            error={"type": "RuntimeError", "message": "crash"},
        )

        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["type"], "RuntimeError")

    def test_record_rejects_absolute_artifact_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be relative"):
            record(
                make_run(),
                {"status": "no_ok_submit"},
                dimensions={},
                timing={},
                artifacts={"workspace": "/tmp/workspace"},
            )


if __name__ == "__main__":
    unittest.main()
