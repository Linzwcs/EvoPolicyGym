"""Run-level driver that connects a host to an agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..agent import Launch, Loop, Transcript
from ..infra.http import Server
from .local import Host

Json = Any


@dataclass(frozen=True, slots=True)
class Trial:
    """Result of one driven EvoPolicyGym run."""

    host: Host
    launch: Launch
    transcript: Transcript

    @property
    def done(self) -> bool:
        return self.transcript.done and not self.host.run.alive()


@dataclass(frozen=True, slots=True)
class Drive:
    """Serve one local host and drive one persistent agent loop."""

    loop: Loop
    bind: str = "127.0.0.1"
    port: int = 0
    env: Mapping[str, str] = field(default_factory=dict)

    def run(self, host: Host) -> Trial:
        _emit(host, "drive.start", bind=self.bind, port=self.port)
        server = Server(host.service, host=self.bind, port=self.port)
        try:
            with server:
                url = server.url
                _emit(host, "server.start", url=url)
                launch = Launch.from_host(host, url, env=self.env)
                _emit(host, "loop.start", workspace=str(launch.workspace))
                transcript = self.loop.run(launch, done=lambda: not host.run.alive())
                _emit(
                    host,
                    "loop.finish",
                    reason=transcript.reason,
                    session=transcript.session,
                    replies=len(transcript.replies),
                    run_alive=host.run.alive(),
                )
            _emit(host, "server.stop", url=url)
            trial = Trial(host=host, launch=launch, transcript=transcript)
            _emit(host, "drive.finish", done=trial.done)
            return trial
        finally:
            release = getattr(host.store, "release_lock", None)
            if callable(release):
                release()


def drive(
    host: Host,
    loop: Loop,
    *,
    bind: str = "127.0.0.1",
    port: int = 0,
    env: Mapping[str, str] | None = None,
) -> Trial:
    """Convenience wrapper for `Drive(loop).run(host)`."""

    return Drive(loop=loop, bind=bind, port=port, env=dict(env or {})).run(host)


def _emit(host: Host, event: str, **data: Json) -> None:
    emit = getattr(host.store, "emit", None)
    if callable(emit):
        emit(event, **data)
