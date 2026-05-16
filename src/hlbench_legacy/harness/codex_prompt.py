"""Build a Codex prompt for one HL improvement step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hlbench.harness.context import build_context


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_TEMPLATE = REPO_ROOT / "prompts" / "codex_harness_step.md"


def build_prompt(scenario: str, run_dir: Path | None) -> str:
    context = build_context(scenario, run_dir)
    return (
        PROMPT_TEMPLATE.read_text()
        + "\n\n## Current Learner Context\n\n"
        + "```json\n"
        + json.dumps(context, indent=2)
        + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = build_prompt(args.scenario, args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt)
    print(args.output)


if __name__ == "__main__":
    main()

