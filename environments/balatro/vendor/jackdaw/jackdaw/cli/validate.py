"""Scenario-based validation — compare jackdaw engine against live Balatro.

Usage::

    jackdaw validate                          # run all scenarios
    jackdaw validate --category jokers        # run only joker scenarios
    jackdaw validate --scenario joker_joker   # run one scenario
"""

from __future__ import annotations


def run_validate(
    category: str | None = None,
    scenario: str | None = None,
    host: str = "127.0.0.1",
    port: int = 12346,
    delay: float = 0.3,
) -> int:
    """Run validation scenarios against a live balatrobot instance.

    Returns 0 if all scenarios pass, 1 otherwise.
    """
    from jackdaw.bridge.backend import LiveBackend, RPCError, SimBackend
    from jackdaw.cli.scenarios import ScenarioResult, get_scenarios

    # Connect to balatrobot
    live_backend = LiveBackend(host=host, port=port)
    try:
        live_backend.handle("health", None)
    except Exception as e:
        print(f"Cannot reach balatrobot at http://{host}:{port}: {e}")
        print("Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>")
        return 1

    # Collect scenarios
    scenarios = get_scenarios(category=category, name=scenario)
    if not scenarios:
        if scenario:
            print(f"No scenario found with name: {scenario}")
        elif category:
            print(f"No scenarios found in category: {category}")
        else:
            print("No scenarios registered")
        return 1

    print(f"Running {len(scenarios)} scenario(s)...")
    if category:
        print(f"  Category: {category}")
    if scenario:
        print(f"  Scenario: {scenario}")
    print()

    # Run scenarios
    results: list[tuple[str, str, ScenarioResult]] = []
    passed = 0
    failed = 0

    for s in scenarios:
        sim = SimBackend()
        print(f"  [{s.category}] {s.name}: {s.description}")

        try:
            result = s.run(sim.handle, live_backend.handle, delay=delay)
        except RPCError as e:
            result = ScenarioResult(
                passed=False,
                diffs=[f"RPCError: {e.message} (code={e.code})"],
                details=f"RPC error: {e.message}",
            )
        except Exception as e:
            result = ScenarioResult(
                passed=False,
                diffs=[f"Exception: {e}"],
                details=f"Error: {e}",
            )

        results.append((s.name, s.category, result))

        if result.sub_results:
            for sub_name, sub_result in result.sub_results:
                if sub_result.passed:
                    passed += 1
                    print(f"    [{sub_name}] PASS  {sub_result.details}")
                else:
                    failed += 1
                    print(f"    [{sub_name}] FAIL")
                    for d in sub_result.diffs:
                        print(f"      {d}")
        elif result.passed:
            passed += 1
            print("    PASS")
        else:
            failed += 1
            print("    FAIL")
            for d in result.diffs:
                print(f"      {d}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Group by category — expand sub_results for per-item reporting
    categories: dict[str, list[tuple[str, ScenarioResult]]] = {}
    for name, cat, result in results:
        if result.sub_results:
            for sub_name, sub_result in result.sub_results:
                categories.setdefault(cat, []).append((sub_name, sub_result))
        else:
            categories.setdefault(cat, []).append((name, result))

    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for _, r in cat_results if r.passed)
        cat_total = len(cat_results)
        status = "PASS" if cat_passed == cat_total else "FAIL"
        print(f"\n  {cat}: {status} ({cat_passed}/{cat_total})")
        for name, r in cat_results:
            mark = "+" if r.passed else "X"
            print(f"    [{mark}] {name}")

    print(f"\nTotal: {passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1
