"""Data shapes shared by the decomposer, the rules, and the UI.

Everything here is built from what the proxy already measures and what Ollama's
own final chunk reports - no process or GPU monitoring exists yet, so no finding
claims a cause that would need it (see rules.py for how that boundary is kept).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# How sure a finding is that it has named the real cause.
CONFIRMED = "confirmed"   # directly measured
LIKELY = "likely"         # a strong, specific inference
POSSIBLE = "possible"     # named for completeness; could be wrong

_LABELS = {
    "connect": "Connecting",
    "queue": "Waiting in line",
    "load": "Loading the model",
    "prompt": "Reading the prompt",
    "generate": "Generating",
    "client_wait": "Waiting on the app",
    "other": "Other",
}


@dataclass(slots=True)
class Segment:
    """One slice of the request's wall-clock time."""
    kind: str          # connect | queue | load | prompt | generate | client_wait | other
    ns: int
    measured: bool      # False when inferred rather than read from a direct measurement

    @property
    def label(self) -> str:
        return _LABELS.get(self.kind, self.kind.capitalize())


@dataclass(slots=True)
class Breakdown:
    """A Trace's wall time, split into named segments that sum to total_ns."""
    total_ns: int
    segments: list[Segment]

    @property
    def largest(self) -> Segment | None:
        return max(self.segments, key=lambda s: s.ns, default=None)

    def ns(self, kind: str) -> int:
        return next((s.ns for s in self.segments if s.kind == kind), 0)

    def fraction(self, kind: str) -> float:
        return self.ns(kind) / self.total_ns if self.total_ns > 0 else 0.0


@dataclass(slots=True)
class Evidence:
    text: str
    measured: bool = True


@dataclass(slots=True)
class Finding:
    code: str
    severity: str            # info | warn | critical
    confidence: str          # CONFIRMED | LIKELY | POSSIBLE
    title: str
    explanation: str
    evidence: list[Evidence] = field(default_factory=list)
    fix: str | None = None
    impact_ns: int | None = None   # how much of the total this explains; ranks findings
    note: str | None = None        # an aside: what would sharpen this if it were available


@dataclass(slots=True)
class Diagnosis:
    breakdown: Breakdown | None
    primary: Finding | None
    also: list[Finding] = field(default_factory=list)