"""Core server-side machinery for hlbench.

Modules:
- server: top-level `Server` class (one instance per run)
- submit_handler: 7-phase submit lifecycle (see docs/submit-protocol.md)
- env_runner: run one episode against a Policy
- sandbox: subprocess isolation for policy execution (also imports
  system/policy.py from the snapshot dir, so there's no separate
  policy_loader module — see sandbox._child_main)
- feedback: write summary.json + per-episode artifacts
- seed_resolver: load train.json / heldout.json, resolve env_instance ID -> real seed
"""
