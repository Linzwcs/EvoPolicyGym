"""Small sanitized public failure hierarchy."""


class EvoPolicyGymError(Exception):
    """Base class for expected EvoPolicyGym failures."""


class ProgramError(EvoPolicyGymError, ValueError):
    """A Program cannot be captured, admitted, or loaded."""


class ProgramSourceError(ProgramError):
    """A Program directory is missing or contains unsupported content."""


class ProgramLimitError(ProgramError):
    """A Program exceeds a configured capture limit."""


class ProgramChangedError(ProgramError):
    """A Program directory changed while it was being captured."""


class BenchmarkError(EvoPolicyGymError):
    """A Benchmark definition or trusted operation failed."""


class EvaluationError(EvoPolicyGymError):
    """Evaluation could not produce a valid public result."""


class AgentRunError(EvoPolicyGymError):
    """A Coding Agent run could not produce a valid terminal result."""


__all__ = [
    "AgentRunError",
    "BenchmarkError",
    "EvaluationError",
    "EvoPolicyGymError",
    "ProgramChangedError",
    "ProgramError",
    "ProgramLimitError",
    "ProgramSourceError",
]
