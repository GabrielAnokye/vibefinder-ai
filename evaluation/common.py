"""Small helpers shared by the experiment scripts: response snapshots and result files."""

import json
import re
from pathlib import Path
from typing import Any, Dict

from src.assistant.models import EXPLICIT, INFERRED
from src.assistant.planner import describe_value

RESULTS_DIR = Path(__file__).parent / "results"
CITATION = re.compile(r"\[([a-z_]+#[a-z0-9-]+)\]")


def snapshot(response) -> Dict[str, Any]:
    """The observable facts of one assistant response, for side-by-side comparison."""
    prefs = response.preferences or (response.interpretation.prefs if response.interpretation else {})
    plan = response.plan
    explanation = response.explanation
    return {
        "status": response.status,
        "retrieved": [r.passage.source_id for r in response.retrieved],
        "explicit": {f: describe_value(f, p.value) for f, p in prefs.items() if p.provenance == EXPLICIT},
        "inferred": {f: f"{describe_value(f, p.value)} <- {p.source}" for f, p in prefs.items()
                     if p.provenance == INFERRED},
        "hints_not_used": {h.field: h.note for h in plan.hints if not h.used} if plan else {},
        "mode": plan.mode if plan and plan.status == "ok" else None,
        "advanced": plan.advanced if plan and plan.status == "ok" else None,
        "top_ids": [x.song["id"] for x in response.recommendations],
        "distinct_genres": len({x.song["genre"] for x in response.recommendations}),
        "limitations": list(plan.limitations) if plan else [],
        # sources actually cited inside a limitation or no-match message, e.g. "[catalog#genre-coverage]"
        "cited_in_limitations": sorted(set(CITATION.findall(" ".join((plan.limitations if plan else [])
                                                                       + [response.message or ""])))),
        "explanation_citations": sorted({c for ids in explanation.citations.values() for c in ids}) if explanation else [],
        "evidence_sources": ([s["id"] for s in response.payload["sources"]] if response.payload
                             else [r.passage.source_id for r in response.retrieved]),
        "message": response.message,
    }


def write_results(name: str, data: Dict[str, Any], markdown: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS_DIR / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {RESULTS_DIR / name}.json and .md")


def cell(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items()) or "-"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "-"
    return "-" if value is None else str(value).replace("|", "/")
