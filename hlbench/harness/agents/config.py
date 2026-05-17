"""Agent backend configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AgentConfig:
    backend: str = "command"
    name: str = "none"
    command: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "name": self.name,
            "command": list(self.command),
        }


AGENT_PRESETS: dict[str, AgentConfig] = {
    "none": AgentConfig(name="none", backend="command", command=()),
    "codex": AgentConfig(name="codex", backend="command", command=("codex", "exec")),
    "claude": AgentConfig(name="claude", backend="command", command=("claude",)),
}


def get_agent_preset(name: str) -> AgentConfig:
    try:
        return AGENT_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent preset {name!r}; expected one of {sorted(AGENT_PRESETS)}") from exc


def resolve_agent_config(
    *,
    backend: str = "command",
    preset: str = "none",
    command: list[str] | tuple[str, ...] | None = None,
) -> AgentConfig:
    if backend != "command":
        raise ValueError("only the command agent backend is implemented")
    config = replace(get_agent_preset(preset), backend=backend)
    if command is not None:
        return replace(config, name="custom" if preset == "none" else preset, command=tuple(command))
    return config
