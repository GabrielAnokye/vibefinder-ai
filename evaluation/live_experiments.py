"""
Optional live-model experiments. Select a provider and confirm with --yes for external calls.

    python -m evaluation.live_experiments plan                 # show what would run; no requests
    python -m evaluation.live_experiments smoke --provider gemini --yes
    python -m evaluation.live_experiments retrieval --provider gemini --yes
    python -m evaluation.live_experiments prompts --provider gemini --yes
    python -m evaluation.live_experiments prompts --mock       # mechanics only, SIMULATED, no network

Controlled design: for each request the offline workflow builds the plan, ranked songs, and
grounding payload ONCE. The model then sees that identical payload in every arm; only the
variable under test changes (retrieved passages present/withheld, or the prompt style). Model id,
schema, max_tokens and temperature are the same in every arm. Retrieval scores the first answer
only. Smoke and prompt runs also capture the workflow's correction and final response.

Retrieval arms allow two HTTP attempts. Smoke and prompt arms share the workflow's three-attempt
cap across generation, transport retries and correction. See the plan command for totals.
Gemini results and traces go to evaluation/{results,traces}/live_gemini_<experiment>.json.

Metrics are automatic and partly heuristic (keyword checks). Unsupported claims in the prose still
need human review; mocked runs do not measure real model behaviour.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import date
from typing import Any, Dict, List, Optional

from evaluation.common import RESULTS_DIR
from evaluation.generate_traces import TRACE_DIR
from src.assistant import Assistant, MockProvider
from src.assistant.providers import (
    DEFAULT_PROMPT_STYLE, LIVE_PROVIDERS, MAX_HTTP_ATTEMPTS, AttemptBudget, GenerationRequest, ProviderError,
    make_provider, template_explanation,
)
from src.assistant.trace import Trace, redact, sanitized_json
from src.assistant.validation import _strip_fences, parse_explanation

REQUESTS = [
    "Music for studying.",
    "I need energetic music for a workout, preferably rock or hip-hop.",
    "Recommend acoustic music with very high energy.",
]
INITIAL_REQUESTS = {"smoke": 1, "retrieval": 2 * len(REQUESTS), "prompts": 2 * len(REQUESTS)}
RETRIEVAL_HTTP_ATTEMPTS = 2
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
        valid_json = True
    except (ValueError, RecursionError):
        data = None
        valid_json = False
    data = data if isinstance(data, dict) else {}
    raw_items = data.get("items")
    raw_items = raw_items if isinstance(raw_items, list) else []
    results = payload["results"]
    expected_ids = {r["song_id"] for r in results}
    known_items = [item for item in raw_items if isinstance(item, dict)
                   and type(item.get("song_id")) is int and item["song_id"] in expected_ids]
    items = {item["song_id"]: item["explanation"] for item in known_items
             if isinstance(item.get("explanation"), str) and item["explanation"].strip()}
    allowed = {s["id"] for s in payload["sources"]}
    cited = [citation for item in raw_items if isinstance(item, dict)
             and isinstance(item.get("citations"), list) for citation in item["citations"]]
    limitations = data.get("limitations")

    def prose(song_id):
        return items.get(song_id, "").casefold()

    with_unmet = [r for r in results if r["unmet"]]
    with_inferred = [r for r in results if r["inferred_used"]]
    return {
        "valid_json": valid_json,
        "schema_valid": explanation is not None,
        "errors": errors,
        "coverage": round(len(items) / len(results), 3) if results else None,
        "song_reference_validity": round(len(known_items) / len(raw_items), 3) if raw_items else None,
        "citations": len(cited),
        "citation_validity": (round(sum(isinstance(c, str) and c in allowed for c in cited) / len(cited), 3)
                              if cited else None),
        "unmet_disclosed (heuristic)": (round(sum(all(_keyword(u) in prose(r["song_id"]) for u in r["unmet"])
                                                  for r in with_unmet) / len(with_unmet), 3) if with_unmet else None),
        "feature_mentioned (heuristic)": (round(sum(any(w in prose(r["song_id"]) for w in FEATURE_WORDS)
                                                    for r in results) / len(results), 3) if results else None),
        "inferred_distinguished (heuristic)": (round(sum(any(w in prose(r["song_id"]) for w in INFERRED_WORDS)
                                                         for r in with_inferred) / len(with_inferred), 3)
                                               if with_inferred else None),
        "limitations_present (heuristic)": (isinstance(limitations, list) and
                                            any(isinstance(x, str) and x.strip() for x in limitations)
                                            if payload["limitations"] else None),
        "mean_words_per_item": (round(sum(len(item.split()) for item in items.values()) / len(items), 1)
                                if items else None),
        "unsupported_claims": "pending human review",
    }


class RecordingProvider:
    """Capture each logical call while letting the assistant decide when to correct or fall back."""

    def __init__(self, provider):
        self.provider = provider
        self.name, self.label = provider.name, provider.label
        self.answers: List[Dict[str, Any]] = []

    def generate(self, request: GenerationRequest, budget: AttemptBudget) -> str:
        before = budget.used
        answer = {"stage": "correction" if request.is_correction else "initial", "raw": None, "error": None}
        try:
            answer["raw"] = self.provider.generate(request, budget)
            return answer["raw"]
        except ProviderError as error:
            answer["error"] = f"{error.kind}: {error}"
            answer["error_kind"] = error.kind
            answer["raw"] = error.partial_output
            raise
        finally:
            answer["http_attempts"] = budget.used - before
            answer["response_metadata"] = request.response_metadata.copy()
            self.answers.append(answer)


def run_record(arm, payload, style, recorder, assistant, final=None):
    for answer in recorder.answers:
        metrics = answer["metrics"] = explanation_metrics(answer["raw"], payload, assistant.songs)
        if answer["error"]:
            metrics["schema_valid"] = False
            metrics["errors"].insert(0, answer["error"])
    initial, *corrections = recorder.answers
    source = "simulated" if recorder.name == "mock" else "live"
    if final:
        explanation, generator, status, attempts = final
        if generator == recorder.label and explanation:
            outcome = f"{source}_validated"
            metrics = recorder.answers[-1]["metrics"]
        else:
            outcome = f"{source}_template_fallback"
            raw = json.dumps(template_explanation(payload)) if explanation else None
            metrics = explanation_metrics(raw, payload, assistant.songs)
        final = {"generator": generator, "status": status, "accepted": explanation is not None,
                 "explanation": asdict(explanation) if explanation else None, "metrics": metrics}
    else:
        attempts = initial["http_attempts"]
        outcome = f"{source}_raw_"
        outcome += ("provider_failure" if initial["error"] else
                    "validated" if initial["metrics"]["schema_valid"] else "rejected")
    return {"request": payload["request"], "arm": arm, "style": style, "grounding": payload,
            "initial": initial, "correction": corrections[0] if corrections else None, "final": final,
            "outcome": outcome, "logical_requests": len(recorder.answers), "http_attempts": attempts}


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
    runs = []
    trace = None
    if name == "smoke":
        recorder = RecordingProvider(provider)
        response = assistant.ask(REQUESTS[0], provider=recorder)
        if not recorder.answers:
            raise RuntimeError(f"smoke test stopped before reaching the provider: {response.status}")
        final = (response.explanation, response.generator, response.generation_status, response.http_attempts)
        runs.append(run_record(DEFAULT_PROMPT_STYLE, response.payload, DEFAULT_PROMPT_STYLE, recorder, assistant, final))
        response.trace.metadata["outcome"] = runs[0]["outcome"]
        trace = response.trace.to_dict()
    else:
        traces = []
        for payload in fixed_payloads(assistant):
            if name == "retrieval":
                withheld = {**payload, "sources": []}
                arms = [("sources supplied", payload, DEFAULT_PROMPT_STYLE),
                        ("sources withheld", withheld, DEFAULT_PROMPT_STYLE)]
            else:
                arms = [("general prompt", payload, "general"), ("specialized prompt", payload, "specialized")]
            for arm, arm_payload, style in arms:
                recorder = RecordingProvider(provider)
                arm_trace = Trace(request=payload["request"], arm=arm, provider=provider.label, prompt_style=style,
                                  provider_kind="simulated" if provider.name == "mock" else "live")
                final = None
                if name == "retrieval":
                    try:
                        recorder.generate(GenerationRequest(arm_payload, style), AttemptBudget(RETRIEVAL_HTTP_ATTEMPTS))
                    except ProviderError:
                        pass  # The failed arm is kept in the results.
                else:
                    final = assistant.explain(arm_payload, recorder, style, trace=arm_trace)
                run = run_record(arm, arm_payload, style, recorder, assistant, final)
                if name == "retrieval":
                    arm_trace.add("measure_initial", "measure the first answer",
                                  sources=[source["id"] for source in arm_payload["sources"]],
                                  result={"response_metadata": run["initial"]["response_metadata"],
                                          "errors": run["initial"]["metrics"]["errors"]},
                                  validation="accepted" if run["initial"]["metrics"]["schema_valid"] else "rejected")
                arm_trace.add("return", "record experimental result", result={
                    "outcome": run["outcome"], "http_attempts": run["http_attempts"],
                    "generator": run["final"]["generator"] if run["final"] else None,
                })
                runs.append(run)
                traces.append(arm_trace.to_dict())
        trace = {"experiment": name, "runs": traces}
    reported_models = sorted({answer["response_metadata"]["model"] for run in runs
                              for answer in (run["initial"], run["correction"])
                              if answer and "model" in answer["response_metadata"]})
    model_mismatch = any(answer and answer.get("error_kind") == "model_mismatch" for run in runs
                         for answer in (run["initial"], run["correction"]))
    return {"experiment": name, "provider": provider.label, "simulated": provider.name == "mock",
            "requested_model": getattr(provider, "model", None), "reported_models": reported_models,
            "model_consistent": not model_mismatch and len(reported_models) <= 1,
            "generation_settings": {"max_tokens": getattr(provider, "max_tokens", None),
                                    "temperature": getattr(provider, "temperature", None),
                                    "thinking_config": getattr(provider, "thinking_config", None)},
            "date": date.today().isoformat(), "requests": REQUESTS if name != "smoke" else REQUESTS[:1],
            "logical_requests": sum(run["logical_requests"] for run in runs),
            "http_attempts": sum(run["http_attempts"] for run in runs),
            "trace": trace, "runs": runs}


def _simulated_provider() -> MockProvider:
    return MockProvider([lambda req: json.dumps(template_explanation(req.payload))] * sum(INITIAL_REQUESTS.values()))


def save_evidence(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitized_json(data), encoding="utf-8")
    print(f"wrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Optional live-model experiments (spend credits only with --yes)")
    parser.add_argument("experiment", choices=["plan", "smoke", "retrieval", "prompts", "all"])
    parser.add_argument("--provider", choices=LIVE_PROVIDERS, help="required for live execution")
    parser.add_argument("--yes", action="store_true", help="confirm external requests")
    parser.add_argument("--mock", action="store_true", help="run the mechanics with a SIMULATED provider")
    args = parser.parse_args(argv)
    names = ["smoke", "retrieval", "prompts"] if args.experiment in ("all", "plan") else [args.experiment]
    initial = sum(INITIAL_REQUESTS[n] for n in names)
    corrections = sum(INITIAL_REQUESTS[n] for n in names if n != "retrieval")
    http_attempts = sum(INITIAL_REQUESTS[n] * (RETRIEVAL_HTTP_ATTEMPTS if n == "retrieval" else MAX_HTTP_ATTEMPTS)
                        for n in names)
    print(f"Planned: {', '.join(names)} -> {initial} initial logical requests, up to {corrections} corrections "
          f"({initial + corrections} total logical requests), at most {http_attempts} HTTP attempts.")
    if args.experiment == "plan":
        return 0
    if args.mock:
        provider = _simulated_provider()
        print("SIMULATED provider: scripted template output. These numbers do not measure a real model.")
    else:
        if args.provider is None:
            print("Live experiment pending: select --provider gemini or --provider openrouter and add --yes.")
            return 0
        provider, note = make_provider(args.provider)
        if note:
            print(f"Live experiment pending: {note}.")
            return 0
        if not args.yes:
            print(redact(f"Not run: add --yes to send these requests to {provider.label}."))
            return 0
    for name in names:
        result = run_experiment(name, provider)
        for run in result["runs"]:
            initial = run["initial"]["metrics"]
            correction = run["correction"]
            final = run["final"]
            print(f"[{name}] {run['request'][:45]:<45} {run['arm']:<20} "
                  f"initial_valid={initial['schema_valid']} "
                  f"correction_valid={correction['metrics']['schema_valid'] if correction else None} "
                  f"final_accepted={final['accepted'] if final else None} outcome={run['outcome']}")
            if final:
                print(redact(f"  Final generator: {final['generator']}"))
        print(f"Logical requests: {result['logical_requests']}; "
              f"{'simulated ' if args.mock else ''}HTTP attempts: {result['http_attempts']}")
        if not args.mock:
            trace = result.pop("trace")
            filename = f"live_{name}.json" if args.provider == "openrouter" else f"live_{args.provider}_{name}.json"
            save_evidence(TRACE_DIR / filename, trace)
            save_evidence(RESULTS_DIR / filename, result)
        if not result["model_consistent"]:
            print("The provider reported a model change. These runs cannot establish a same-model comparison.")
            return 1
        if name == "smoke" and not result["runs"][0]["outcome"].endswith("_validated"):
            print("Smoke did not produce a validated provider explanation.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
