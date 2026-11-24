"""
Automated evaluation harness for the integrated VibeFinder AI assistant.

Runs every case in evaluation/cases.json through Assistant.ask() (the real workflow), applies
universal checks plus the case's independently written expectations, and prints PASS/FAIL per
case with the failed checks. The pass rate measures agreement with these expectations; it is
NOT a measure of real-world recommendation accuracy.

    python -m evaluation.run_evaluation                      # offline (template + scripted mocks)
    python -m evaluation.run_evaluation --json evaluation/results/offline_evaluation.json
    python -m evaluation.run_evaluation --live --provider gemini --yes

Exit status is 1 when a required case fails.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.assistant import Assistant, MockProvider
from src.assistant.providers import LIVE_PROVIDERS, MAX_HTTP_ATTEMPTS, ProviderError, make_provider, template_explanation
from src.assistant.trace import redact, sanitized_json
from src.recommender import normalize_label, score_song

CASES_PATH = Path(__file__).parent / "cases.json"
LIVE_CASES = ("study-calm", "multiple-genres", "nostalgic-acoustic", "acoustic-high-energy", "injection")


# ---------------------------------------------------------------- scripted mock behaviours (simulated only)

def _valid(req):
    return json.dumps(template_explanation(req.payload))


def _bad_citation(req):
    data = template_explanation(req.payload)
    data["items"][0]["citations"] = ["invented#source"]
    return json.dumps(data)


def _unknown_song(req):
    data = template_explanation(req.payload)
    data["items"][0]["song_id"] = 999
    return json.dumps(data)


MOCK_SCRIPTS: Dict[str, Any] = {
    "valid": _valid,
    "bad_citation": _bad_citation,
    "unknown_song": _unknown_song,
    "invalid_json": "this is not JSON",
    "error:auth": ProviderError("auth", "authentication failed (simulated)"),
    "error:network": ProviderError("network", "timed out (simulated)", retryable=True),
}


def mock_for(script: List[str]) -> MockProvider:
    return MockProvider([MOCK_SCRIPTS[step] for step in script])


# ---------------------------------------------------------------- checks

Check = Callable[[Any, Assistant, Any], Tuple[bool, str]]


def _recs(r):
    return r.recommendations


def _limitations(r) -> List[str]:
    items = list(r.plan.limitations) if r.plan else []
    if r.explanation:
        items += r.explanation.limitations
    return items


def _pref(r, field):
    prefs = r.preferences or (r.interpretation.prefs if r.interpretation else {})
    return prefs.get(field)


def check_prefs(r, a, spec) -> Tuple[bool, str]:
    problems = []
    for field, want in spec.items():
        pref = _pref(r, field)
        if pref is None:
            problems.append(f"{field}: not interpreted")
            continue
        if "provenance" in want and pref.provenance != want["provenance"]:
            problems.append(f"{field}: provenance {pref.provenance}, expected {want['provenance']}")
        value = pref.value
        if "min" in want and not (isinstance(value, (int, float)) and value >= want["min"]):
            problems.append(f"{field}: {value!r} < {want['min']}")
        if "max" in want and not (isinstance(value, (int, float)) and value <= want["max"]):
            problems.append(f"{field}: {value!r} > {want['max']}")
        if "equals" in want and value != want["equals"]:
            problems.append(f"{field}: {value!r} != {want['equals']!r}")
        if "contains" in want and not set(want["contains"]) <= set(value or ()):
            problems.append(f"{field}: {value!r} lacks {want['contains']}")
    return not problems, "; ".join(problems)


def _all(predicate, recs, describe) -> Tuple[bool, str]:
    bad = [describe(x) for x in recs if not predicate(x)]
    return not bad, ("violations: " + ", ".join(bad)) if bad else ""


def _genre(x):
    return normalize_label(x.song["genre"])


EXPECTATIONS: Dict[str, Check] = {
    "status": lambda r, a, v: (r.status == v, f"status {r.status}"),
    "count": lambda r, a, v: (len(_recs(r)) == v, f"{len(_recs(r))} results"),
    "count_max": lambda r, a, v: (len(_recs(r)) <= v, f"{len(_recs(r))} results"),
    "contexts": lambda r, a, v: (set(v) <= set(r.interpretation.context_names),
                                 f"contexts {r.interpretation.context_names}"),
    "prefs": check_prefs,
    "plan": lambda r, a, v: (all(getattr(r.plan, k) == want for k, want in v.items()),
                             ", ".join(f"{k}={getattr(r.plan, k)}" for k in v)),
    "feature_min": lambda r, a, v: _all(lambda x: all(x.song[f] >= m for f, m in v.items()), _recs(r),
                                        lambda x: f"{x.song['title']}"),
    "feature_max": lambda r, a, v: _all(lambda x: all(x.song[f] <= m for f, m in v.items()), _recs(r),
                                        lambda x: f"{x.song['title']}"),
    "genres_only": lambda r, a, v: _all(lambda x: _genre(x) in v, _recs(r), _genre),
    "genres_exclude": lambda r, a, v: _all(lambda x: _genre(x) not in v, _recs(r), _genre),
    "genres_include": lambda r, a, v: (set(v) <= {_genre(x) for x in _recs(r)},
                                       f"genres {sorted({_genre(x) for x in _recs(r)})}"),
    "artists_exclude": lambda r, a, v: _all(lambda x: x.song["artist"] not in v, _recs(r), lambda x: x.song["artist"]),
    "unique_artists": lambda r, a, v: (len({x.song["artist"] for x in _recs(r)}) == len(_recs(r)),
                                       f"artists {[x.song['artist'] for x in _recs(r)]}"),
    "song_ids_absent": lambda r, a, v: _all(lambda x: x.song["id"] not in v, _recs(r), lambda x: str(x.song["id"])),
    "all_unmet_contains": lambda r, a, v: _all(lambda x: v in x.unmet, _recs(r), lambda x: x.song["title"]),
    "limitation_contains": lambda r, a, v: (all(any(s in item for item in _limitations(r)) for s in v),
                                            f"limitations {_limitations(r)}"),
    "message_contains": lambda r, a, v: (all(s in (r.message or "") for s in v), f"message {r.message!r}"),
    "generator_contains": lambda r, a, v: (v in r.generator, f"generator {r.generator!r}"),
    "generation_status_contains": lambda r, a, v: (v in r.generation_status, f"status {r.generation_status!r}"),
    "http_attempts_max": lambda r, a, v: (r.http_attempts <= v, f"{r.http_attempts} attempts"),
}


def universal_checks(r, a: Assistant) -> List[Tuple[str, bool, str]]:
    """Checks every answered request must pass, whatever it asked for."""
    checks = [("trace_complete", bool(r.trace and r.trace.stages()[-1] == "return"), "")]
    if r.status != "ok":
        checks.append(("no_recommendations_unless_ok", not r.recommendations, f"{len(r.recommendations)} results"))
        return checks
    recs = r.recommendations
    in_catalog = all(a.songs_by_id.get(x.song["id"]) == x.song for x in recs)
    consistent = all(math.isfinite(x.score) and
                     x.score == score_song(r.plan.engine_prefs, x.song, r.plan.mode, r.plan.advanced)[0]
                     for x in recs)
    allowed = {s["id"] for s in r.payload["sources"]} if r.payload else set()
    cited = {c for ids in (r.explanation.citations.values() if r.explanation else []) for c in ids}
    checks += [
        ("all_in_catalog", in_catalog, ""),
        ("scores_match_engine", consistent, ""),
        ("within_requested_count", len(recs) <= r.plan.k, f"{len(recs)} > {r.plan.k}"),
        ("citations_valid", cited <= allowed and allowed <= set(a.retriever.by_id), f"cited {sorted(cited - allowed)}"),
        ("explanation_validated", r.explanation is not None and "failed validation" not in r.generation_status,
         r.generation_status),
    ]
    return checks


# ---------------------------------------------------------------- running

def load_cases(path: Path = CASES_PATH) -> List[Dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    for case in cases:
        unknown = set(case["expect"]) - set(EXPECTATIONS)
        if unknown or "request" not in case:
            raise ValueError(f"case {case['id']}: bad definition (unknown expectations {sorted(unknown)})")
    return cases


def evaluate_case(assistant: Assistant, case: Dict[str, Any], provider=None) -> Dict[str, Any]:
    simulated = "mock" in case
    if simulated:
        provider = mock_for(case["mock"])
    try:
        response = assistant.ask(case["request"], provider=provider, retrieval=case.get("retrieval", True))
    except Exception as error:  # a crash is a failed case, not a crashed harness
        checks = [{"name": "no_crash", "passed": False, "detail": f"{type(error).__name__}: {error}"}]
        response = None
    else:
        checks = [{"name": "no_crash", "passed": True, "detail": ""}]
        checks += [{"name": n, "passed": p, "detail": "" if p else d} for n, p, d in universal_checks(response, assistant)]
        for name, want in case["expect"].items():
            try:
                passed, detail = EXPECTATIONS[name](response, assistant, want)
            except Exception as error:
                passed, detail = False, f"check error: {type(error).__name__}: {error}"
            checks.append({"name": name, "passed": bool(passed), "detail": "" if passed else detail})
    result = {
        "id": case["id"], "name": case.get("name", case["id"]), "required": case.get("required", True),
        "simulated": simulated, "status": response.status if response else "crash",
        "song_ids": [x.song["id"] for x in response.recommendations] if response else [],
        "generator": response.generator if response else "",
        "passed": all(c["passed"] for c in checks), "checks": checks,
    }
    if provider is not None and not simulated:
        result["generation"] = {
            "accepted": bool(response and response.explanation and response.generator == provider.label),
            "template_fallback": bool(response and response.generator.startswith("template")),
            "http_attempts": response.http_attempts if response else 0,
            "status": response.generation_status if response else None,
        }
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = [c for r in results for c in r["checks"]]
    passed_cases = sum(r["passed"] for r in results)
    return {
        "cases": len(results), "cases_passed": passed_cases,
        "case_pass_rate": round(passed_cases / len(results), 4) if results else 0.0,
        "checks": len(checks), "checks_passed": sum(c["passed"] for c in checks),
        "check_pass_rate": round(sum(c["passed"] for c in checks) / len(checks), 4) if checks else 0.0,
        "required_failures": [r["id"] for r in results if r["required"] and not r["passed"]],
        "simulated_cases": [r["id"] for r in results if r["simulated"]],
    }


def print_report(title: str, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print(redact(title))
    print("=" * len(title))
    for r in results:
        ok = sum(c["passed"] for c in r["checks"])
        tag = " [simulated provider]" if r["simulated"] else ""
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<34} {ok}/{len(r['checks'])} checks{tag}")
        for c in r["checks"]:
            if not c["passed"]:
                print(redact(f"        failed {c['name']}: {c['detail']}"))
    print(f"\nCases: {summary['cases_passed']}/{summary['cases']} passed ({summary['case_pass_rate']:.1%})")
    print(f"Checks: {summary['checks_passed']}/{summary['checks']} passed ({summary['check_pass_rate']:.1%})")
    if summary["required_failures"]:
        print(f"Required failures: {', '.join(summary['required_failures'])}")
    print("This measures agreement with the expectations in evaluation/cases.json, not real-world "
          "recommendation quality.")


def run(cases: List[Dict[str, Any]], assistant: Optional[Assistant] = None, provider=None) -> List[Dict[str, Any]]:
    assistant = assistant or Assistant()
    return [evaluate_case(assistant, case, provider) for case in cases]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the VibeFinder AI assistant")
    parser.add_argument("--cases", default=str(CASES_PATH), help="cases JSON file")
    parser.add_argument("--json", metavar="PATH", help="also write structured results to PATH")
    parser.add_argument("--live", action="store_true",
                        help=f"also run {len(LIVE_CASES)} cases with the selected live provider")
    parser.add_argument("--provider", choices=LIVE_PROVIDERS, help="required with --live for external requests")
    parser.add_argument("--yes", action="store_true", help="confirm live external requests")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    offline = run(cases)
    summary = summarize(offline)
    print_report("Offline evaluation (template provider; mock cases are simulated)", offline, summary)
    output: Dict[str, Any] = {"offline": {"summary": summary, "results": offline}}

    if args.live:
        live_cases = [c for c in cases if c["id"] in LIVE_CASES]
        print(f"\nLive evaluation: {len(live_cases)} cases, "
              f"at most {MAX_HTTP_ATTEMPTS * len(live_cases)} external requests.")
        provider, note = (make_provider(args.provider) if args.provider else
                          (None, "select --provider gemini or --provider openrouter"))
        if note:
            print(f"Live evaluation pending: {note}.")
            output["live"] = {"status": "pending", "reason": note}
        elif not args.yes:
            print("Live evaluation not run: add --yes to confirm the external requests.")
            output["live"] = {"status": "pending", "reason": "not confirmed with --yes"}
        else:
            live = run(live_cases, provider=provider)
            live_summary = summarize(live)
            generation = {
                key: sum(case["generation"][key] for case in live if "generation" in case)
                for key in ("accepted", "template_fallback", "http_attempts")
            }
            print()
            print_report(f"Live evaluation ({provider.label})", live, live_summary)
            print(f"Validated provider explanations: {generation['accepted']}/{len(live_cases)}. "
                  f"Template fallbacks: {generation['template_fallback']}. HTTP attempts: {generation['http_attempts']}.")
            output["live"] = {"provider": provider.label, "summary": live_summary, "generation": generation,
                              "results": live}
            summary["required_failures"] += [f"live:{i}" for i in live_summary["required_failures"]]

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sanitized_json(output, ensure_ascii=True), encoding="utf-8")
        print(f"\nResults written to {path}")
    return 1 if summary["required_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
