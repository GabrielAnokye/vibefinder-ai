"""
Optional live-model experiments (OpenRouter-compatible provider). Nothing runs without --yes.

    python -m evaluation.live_experiments plan                 # show what would run; no requests
    python -m evaluation.live_experiments smoke --yes          # 1 logical request
    python -m evaluation.live_experiments retrieval --yes      # sources supplied vs withheld
    python -m evaluation.live_experiments prompts --yes        # general vs specialized prompt
    python -m evaluation.live_experiments prompts --mock       # mechanics only, SIMULATED, no network

Controlled design: for each request the offline workflow builds the plan, ranked songs, and
grounding payload ONCE. The model then sees that identical payload in every arm; only the
variable under test changes (retrieved passages present/withheld, or the prompt style). Model id,
schema, max_tokens and temperature are the same in every arm. Comparisons measure the FIRST answer
(no content correction), so the arms are not given different numbers of chances.

Every logical request may use one transport retry, so the HTTP count is at most twice the number
of logical requests. Results go to evaluation/results/live_<experiment>.json.

Metrics are automatic and partly heuristic (keyword checks). Unsupported claims in the prose still
need human review; mocked runs do not measure real model behaviour.
"""

import argparse
import copy
import json
import re
import sys
from datetime import date
from typing import Any, Dict, List, Optional

from evaluation.common import RESULTS_DIR
from src.assistant import Assistant, MockProvider
from src.assistant.providers import (
    AttemptBudget, GenerationRequest, ProviderError, make_provider, template_explanation,
)
from src.assistant.validation import _strip_fences, parse_explanation

REQUESTS = [
    "Music for studying.",
    "I need energetic music for a workout, preferably rock or hip-hop.",
    "Recommend acoustic music with very high energy.",
]
LOGICAL_REQUESTS = {"smoke": 1, "retrieval": 2 * len(REQUESTS), "prompts": 2 * len(REQUESTS)}
UNMET_KEYWORDS = {"acoustic sound": "acoustic", "mood tags": "mood", "release decade": "decade",
                  "vocal intensity": "vocal", "instrumentalness": "instrumental", "energy": "energy",
                  "genre": "genre", "mood": "mood", "popularity": "popular"}
FEATURE_WORDS = ("energy", "acoustic", "instrumental", "vocal", "popular", "decade", "tag", "tempo",
                 "genre", "mood")
INFERRED_WORDS = ("inferred", "suggested", "guide")


def _keyword(label: str) -> str:
    match = re.match(r"genre \((.+)\)", label)
    return match.group(1) if match else UNMET_KEYWORDS.get(label, label)


def explanation_metrics(raw: Optional[str], payload: Dict[str, Any], catalog: List[Dict]) -> Dict[str, Any]:
    """Automatic metrics for one raw model answer. Heuristic where noted; prose still needs human review."""
    if raw is None:
        return {"valid_json": False, "schema_valid": False, "errors": ["no answer"]}
    explanation, errors = parse_explanation(raw, payload, catalog)
    try:
        data = json.loads(_strip_fences(raw))
    except ValueError:
        data = None
    items = {}
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = {i["song_id"]: i for i in data["items"]
                 if isinstance(i, dict) and isinstance(i.get("song_id"), int) and isinstance(i.get("explanation"), str)}
    results = payload["results"]
    allowed = {s["id"] for s in payload["sources"]}
    cited = [c for i in items.values() for c in i.get("citations", []) if isinstance(c, str)]

    def prose(song_id):
        return items.get(song_id, {}).get("explanation", "").casefold()

    with_unmet = [r for r in results if r["unmet"]]
    with_inferred = [r for r in results if r["inferred_used"]]
    return {
        "valid_json": isinstance(data, dict),
        "schema_valid": explanation is not None,
        "errors": errors,
        "coverage": round(sum(r["song_id"] in items for r in results) / len(results), 3) if results else None,
        "citations": len(cited),
        "citation_validity": round(sum(c in allowed for c in cited) / len(cited), 3) if cited else None,
        "unmet_disclosed (heuristic)": (round(sum(all(_keyword(u) in prose(r["song_id"]) for u in r["unmet"])
                                                  for r in with_unmet) / len(with_unmet), 3) if with_unmet else None),
        "feature_mentioned (heuristic)": (round(sum(any(w in prose(r["song_id"]) for w in FEATURE_WORDS)
                                                    for r in results) / len(results), 3) if results else None),
        "inferred_distinguished (heuristic)": (round(sum(any(w in prose(r["song_id"]) for w in INFERRED_WORDS)
                                                         for r in with_inferred) / len(with_inferred), 3)
                                               if with_inferred else None),
        "mean_words_per_item": (round(sum(len(i["explanation"].split()) for i in items.values()) / len(items), 1)
                                if items else None),
        "unsupported_claims": "pending human review",
    }


def first_answer(provider, payload: Dict[str, Any], style: str) -> Dict[str, Any]:
    """One logical request (transport retry allowed, no content correction)."""
    budget = AttemptBudget(2)
    try:
        raw = provider.generate(GenerationRequest(payload, style), budget)
        return {"raw": raw, "http_attempts": budget.used, "error": None}
    except ProviderError as error:
        return {"raw": None, "http_attempts": budget.used, "error": f"{error.kind}: {error}"}


def fixed_payloads(assistant: Assistant) -> List[Dict[str, Any]]:
    """Plan, ranking and grounding built once per request by the offline workflow."""
    payloads = []
    for text in REQUESTS:
        response = assistant.ask(text)
        if response.status != "ok":
            raise RuntimeError(f"request did not produce recommendations: {text!r}")
        payloads.append(response.payload)
    return payloads


def run_experiment(name: str, provider, assistant: Optional[Assistant] = None) -> Dict[str, Any]:
    assistant = assistant or Assistant()
    payloads = fixed_payloads(assistant)
    runs = []
    if name == "smoke":
        answer = first_answer(provider, payloads[0], "specialized")
        runs.append({"request": payloads[0]["request"], "arm": "specialized",
                     "http_attempts": answer["http_attempts"], "error": answer["error"],
                     "metrics": explanation_metrics(answer["raw"], payloads[0], assistant.songs),
                     "raw": answer["raw"]})
    for payload in payloads if name in ("retrieval", "prompts") else []:
        if name == "retrieval":
            withheld = copy.deepcopy(payload)
            withheld["sources"] = []
            arms = [("sources supplied", payload, "specialized"), ("sources withheld", withheld, "specialized")]
        else:
            arms = [("general prompt", payload, "general"), ("specialized prompt", payload, "specialized")]
        for arm, arm_payload, style in arms:
            answer = first_answer(provider, arm_payload, style)
            runs.append({"request": payload["request"], "arm": arm, "style": style,
                         "http_attempts": answer["http_attempts"], "error": answer["error"],
                         "metrics": explanation_metrics(answer["raw"], arm_payload, assistant.songs),
                         "raw": answer["raw"]})
    return {"experiment": name, "provider": provider.label, "simulated": provider.name == "mock",
            "date": date.today().isoformat(), "requests": REQUESTS if name != "smoke" else REQUESTS[:1],
            "runs": runs}


def _simulated_provider() -> MockProvider:
    return MockProvider([lambda req: json.dumps(template_explanation(req.payload))] * 20)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Optional live-model experiments (spend credits only with --yes)")
    parser.add_argument("experiment", choices=["plan", "smoke", "retrieval", "prompts", "all"])
    parser.add_argument("--yes", action="store_true", help="confirm paid external requests")
    parser.add_argument("--mock", action="store_true", help="run the mechanics with a SIMULATED provider")
    args = parser.parse_args(argv)
    names = ["smoke", "retrieval", "prompts"] if args.experiment in ("all", "plan") else [args.experiment]
    logical = sum(LOGICAL_REQUESTS[n] for n in names)
    print(f"Planned: {', '.join(names)} -> {logical} logical requests, at most {2 * logical} HTTP requests "
          f"(one transport retry each), no content corrections.")
    if args.experiment == "plan":
        return 0
    if args.mock:
        provider = _simulated_provider()
        print("SIMULATED provider: scripted template output. These numbers do not measure a real model.")
    else:
        provider, note = make_provider("openrouter")
        if note:
            print(f"Live experiment pending: {note}. Configure OPENROUTER_API_KEY and OPENROUTER_MODEL first.")
            return 0
        if not args.yes:
            print(f"Not run: add --yes to send these requests to {provider.label}.")
            return 0
    for name in names:
        result = run_experiment(name, provider)
        for run in result["runs"]:
            m = run["metrics"]
            print(f"[{name}] {run['request'][:45]:<45} {run['arm']:<20} schema_valid={m['schema_valid']} "
                  f"coverage={m.get('coverage')} citation_validity={m.get('citation_validity')} "
                  f"errors={len(m.get('errors', []))}")
        if not args.mock:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = RESULTS_DIR / f"live_{name}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
