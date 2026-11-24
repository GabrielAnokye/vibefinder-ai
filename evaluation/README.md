# Evaluation and evidence

Everything here is reproducible offline with the commands below. Nothing in this folder is a live
LLM response: offline runs use the deterministic **template provider** (not an LLM), and files or
cases labeled **simulated** use a scripted mock provider to exercise validation, correction and
fallback. Live OpenRouter experiments are **pending** (no credentials were configured).

| Evidence | File | Regenerate with |
|---|---|---|
| Evaluation cases (22, expectations written from the requests and catalog facts) | [cases.json](cases.json) | — |
| Evaluation results | [results/offline_evaluation.txt](results/offline_evaluation.txt), [.json](results/offline_evaluation.json) | `python -m evaluation.run_evaluation --json evaluation/results/offline_evaluation.json` |
| Execution traces (7) | [traces/](traces/) | `python -m evaluation.generate_traces` (`--check` verifies) |
| Retrieval on/off and single- vs multi-source | [results/retrieval_comparison.md](results/retrieval_comparison.md) | `python -m evaluation.retrieval_comparison` |
| Quality floor and energy tolerance | [results/quality_floor.md](results/quality_floor.md) | `python -m evaluation.floor_comparison` |
| Demonstration CLI outputs | [results/demo_outputs.md](results/demo_outputs.md) | `python -m evaluation.capture_demos` |
| Live experiments (pending) | `results/live_*.json` once run | `python -m evaluation.live_experiments plan` |

## Traces

| Trace | What it shows | Provider |
|---|---|---|
| [offline_study_success.json](traces/offline_study_success.json) | full successful workflow; studying hints inferred from `listening_contexts#studying` | template |
| [offline_workout_multi_genre.json](traces/offline_workout_multi_genre.json) | two soft genres; energy-focused mode suggested by the workout guide | template |
| [offline_unique_artists.json](traces/offline_unique_artists.json) | hard "no repeated artists" removes a second LoRoom song; fewer results disclosed | template |
| [offline_no_match_only_metal.json](traces/offline_no_match_only_metal.json) | hard constraint cannot be met; stops with no-match, no substitution | template |
| [offline_poor_coverage_acoustic_energy.json](traces/offline_poor_coverage_acoustic_energy.json) | conflicting soft preferences; compromise and missed preference disclosed | template |
| [simulated_invalid_then_corrected.json](traces/simulated_invalid_then_corrected.json) | answer rejected (unsupported citation) → one correction → accepted | **simulated mock** |
| [simulated_invalid_then_fallback.json](traces/simulated_invalid_then_fallback.json) | answer rejected (unknown song id) → correction rejected (invalid JSON) → template | **simulated mock** |

Traces record stages, actions, structured inputs/results, source ids, validation outcomes and
decisions. They contain no model reasoning, HTTP headers, or credentials.

## Results summary (offline, 2026-11-23)

**Evaluation harness.** 22/22 cases and 197/197 checks pass (3 of the cases use a simulated
provider). This shows agreement with the expectations in `cases.json`. It is not a measure of
recommendation quality, and the expectations were written by the same developer-and-agent team
that built the system, so shared blind spots are possible.

**Retrieval on vs off.** For "Music for studying.", "Music for a workout." and "Something for a
quiet evening.", the request names only a situation. With retrieval the guides supply documented
targets (labeled *inferred*) and the assistant answers; without retrieval it asks for
clarification. For "Recommend acoustic music with very high energy." and "Only recommend metal
songs." the outcome is the same, but only the retrieval run cites the catalog fact behind the
limitation. For "I'm studying and want energetic music" the explicit energy (0.85) is kept and
the guide's energy hint is reported as not used. Whether the inferred targets give *better music*
was not measured; it needs listeners.

**Single source (contexts guide, 5 passages) vs all sources (62 passages).**

| request | evidence passages | limitations backed by a catalog source | other difference |
|---|---|---|---|
| acoustic, very high energy | 3 / 4 | 0 / 1 | same ranking |
| some metal | 0 / 2 | 0 / 1 | single source: clarification; all sources: labeled substitutes |
| only metal | 0 / 2 | 0 / 1 | same no-match |
| music for studying | 1 / 3 | 0 / 0 | same ranking; feature passages available to an LLM |
| quiet evening | 1 / 5 | 0 / 0 | same ranking |

The extra sources add evidence and citations for limitations. Only in the "some metal" case do
they change the recommendations. With the template provider, the additional feature passages are
not used in the prose; they would only matter for a live model.

**Quality floor** (share of the points available from the active preferences).

| floor | returned (8 requests) | weak (<50%) | notes |
|---|---|---|---|
| none | 45 | 12 | "Give me 10 happy songs" returns 7 songs that scored 0% |
| 40% (kept) | 36 | 3 | drops the zero-point songs; keeps one-label matches such as Focus Flow (lofi, 57%) and Spacewalk Thoughts (chill, 43%) for "Chill lofi music" |
| 60% | 32 | 0 | also drops Focus Flow and Spacewalk Thoughts, leaving 2 songs for "Chill lofi music" |

40% was kept: no floor presents zero-point songs as recommendations, and 60% removes songs that
match one of two stated labels. The values are heuristics, not learned or calibrated.

**Energy tolerance (±0.20).** It only decides whether a song is reported as missing a stated
energy preference; scores are unaffected. Checking songs near the boundary found a
floating-point bug: Island Time Radio (0.55) against a 0.35 target has a gap of exactly 0.20 but
was reported as a miss because `0.55 - 0.35 = 0.20000000000000007`. The tolerances are now
inclusive. The ±0.20 value itself was kept.

## Material for the model card (to be finalized later)

- **What is measured automatically:** status, extracted preferences and their provenance, hard
  constraints, catalog membership, exact engine scores, counts, unique artists, citation ids
  against retrieved sources, limitation disclosure, correction/fallback behavior, external
  request limits.
- **What still needs human evaluation:** whether generated or template prose is true and helpful;
  whether a cited passage actually supports the sentence citing it; whether inferred context
  targets (for example, studying → energy 0.35) suit real listeners; musical relevance of any
  ranking; the heuristic metrics in `live_experiments.py` (keyword checks for unmet preferences,
  features and "inferred" wording).
- **Known limitations:** rule-based interpretation covers a fixed vocabulary; heuristic floor and
  tolerances; an 18-song synthetic catalog; evaluation expectations written by the builders;
  prompt-injection defenses limit what generated text can do but are not complete; no live model
  results yet.
