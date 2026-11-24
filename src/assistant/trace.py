"""
Observable execution traces.

A trace records what the workflow did: the stage, the action (often a tool call such as the
recommender or the provider), a structured summary of its input and result, source ids,
validation outcomes, and correction/fallback decisions. It never records model reasoning,
HTTP headers, credentials, or environment variables. Serialization is deterministic (sorted
keys, no timestamps), so offline traces are byte-for-byte reproducible.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

TRACE_FORMAT = "vibefinder-trace/1"
SECRET_ENV_VARS = ("OPENROUTER_API_KEY",)


@dataclass
class TraceEvent:
    step: int
    event_id: str
    stage: str
    action: str
    input: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    validation: Optional[str] = None      # e.g. "accepted", "rejected: <reasons>"
    decision: Optional[str] = None        # e.g. "request one correction", "fall back to template"


class Trace:
    def __init__(self, **metadata: Any):
        self.metadata: Dict[str, Any] = {"format": TRACE_FORMAT, **metadata}
        self.events: List[TraceEvent] = []

    def add(self, stage: str, action: str, *, input: Optional[Dict[str, Any]] = None,
            result: Optional[Dict[str, Any]] = None, sources: Iterable[str] = (),
            validation: Optional[str] = None, decision: Optional[str] = None) -> TraceEvent:
        step = len(self.events) + 1
        event = TraceEvent(step, f"{step:02d}-{stage}", stage, action, dict(input or {}), dict(result or {}),
                           list(sources), validation, decision)
        self.events.append(event)
        return event

    def stages(self) -> List[str]:
        return [event.stage for event in self.events]

    def to_dict(self) -> Dict[str, Any]:
        return {"metadata": self.metadata, "events": [asdict(event) for event in self.events]}

    def to_json(self, secrets: Optional[Iterable[str]] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str)
        return redact(text, secrets) + "\n"

    def summary_lines(self) -> List[str]:
        lines = []
        for e in self.events:
            line = f"{e.step:>2}. {e.stage}: {e.action}"
            if e.validation:
                line += f" | validation: {e.validation}"
            if e.decision:
                line += f" | decision: {e.decision}"
            lines.append(line)
        return lines


def redact(text: str, secrets: Optional[Iterable[str]] = None) -> str:
    """Remove any secret values (by default, the configured API key) from serialized text."""
    values = list(secrets) if secrets is not None else [os.environ.get(name, "") for name in SECRET_ENV_VARS]
    for value in values:
        if value and len(value) >= 4:
            text = text.replace(value, "[redacted]")
    return text
