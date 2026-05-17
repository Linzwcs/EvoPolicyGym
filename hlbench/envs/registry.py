"""Environment backend registry."""

from __future__ import annotations

from hlbench.envs.base import EnvironmentBackend
from hlbench.envs.gymnasium_backend import GymnasiumBackend


_BACKENDS: dict[str, EnvironmentBackend] = {
    GymnasiumBackend.name: GymnasiumBackend(),
}


def register_backend(backend: EnvironmentBackend) -> None:
    if not backend.name:
        raise ValueError("environment backend name must be non-empty")
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> EnvironmentBackend:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"unknown environment backend {name!r}; available: {available}") from exc
