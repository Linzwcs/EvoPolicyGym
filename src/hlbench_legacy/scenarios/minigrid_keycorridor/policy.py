class Policy:
    def reset(self, seed: int, task_config: dict) -> None:
        pass

    def act(self, observation: dict, info: dict) -> int:
        # Trivial S_0 policy: repeatedly turn left. This is intentionally weak.
        return 0
