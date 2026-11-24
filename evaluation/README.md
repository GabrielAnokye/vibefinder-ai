# Evaluation and evidence

Template executions use deterministic prose. Simulated executions use scripted responses or fake
HTTP. The saved Gemini smoke and both comparisons contain actual model output. The retained
OpenRouter smoke records a certificate failure and template fallback.
[Phase 4 live evaluation report](results/live_evaluation_report.md)

| Evidence | File | Regenerate with |
|---|---|---|
| Evaluation cases (22, expectations written from the requests and catalog facts) | [cases.json](cases.json) | — |
| Evaluation results | [results/offline_evaluation.txt](results/offline_evaluation.txt), [.json](results/offline_evaluation.json) | `python -m evaluation.run_evaluation --json evaluation/results/offline_evaluation.json` |
| Offline and simulated traces (7) | [traces/](traces/) | `python -m evaluation.generate_traces` (`--check` verifies) |
| Retrieval on/off and single- vs multi-source | [results/retrieval_comparison.md](results/retrieval_comparison.md) | `python -m evaluation.retrieval_comparison` |
| Quality floor and energy tolerance | [results/quality_floor.md](results/quality_floor.md) | `python -m evaluation.floor_comparison` |
| Demonstration CLI outputs | [results/demo_outputs.md](results/demo_outputs.md) | `python -m evaluation.capture_demos` |
| Phase 4 regression and dry-run checks | [results/phase4_verification.txt](results/phase4_verification.txt) | Commands included in the record |
| Gemini integration verification | [results/gemini_integration_verification.txt](results/gemini_integration_verification.txt) | Mocked Gemini requests, offline tests and byte comparisons |
| Gemini live experiments | [Report](results/live_evaluation_report.md), [smoke JSON](results/live_gemini_smoke.json), [retrieval JSON](results/live_gemini_retrieval.json), [prompt JSON](results/live_gemini_prompts.json) | [Approved live commands](#setup-and-request-budget) |
| Final submission evidence | [Rubric audit and verification](../docs/submission_checklist.md) | [Offline setup](../README.md#getting-started) |

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
| [live_gemini_smoke.json](traces/live_gemini_smoke.json) | end-to-end validation on the first HTTP attempt | **live Gemini** |
| [live_gemini_retrieval.json](traces/live_gemini_retrieval.json) | six first answers, including the rejected study answer with passages withheld | **live Gemini** |
| [live_gemini_prompts.json](traces/live_gemini_prompts.json) | six validated explanations across both prompt styles | **live Gemini** |
| [live_smoke.json](traces/live_smoke.json) | certificate verification failure and safe fallback | **OpenRouter attempt → template** |

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

## Live provider evaluation

### Measured Gemini results

All 13 logical requests in the successful batch took one HTTP attempt each. Gemini reported
`gemini-3.5-flash-lite` throughout. **12/13** initial answers passed application validation.
The smoke passed **1/1**, retrieval passed **3/3** with passages and **2/3** without them, and
each prompt arm passed **3/3**. The rejected study answer cited a guide visible in fixed
provenance whose passage had been withheld. The smoke and prompt runs used no corrections or
template fallbacks. Specialized prompting increased provenance wording and explicit item-level
preference-miss disclosure in the qualitative review, with longer explanations and fewer
structured citations. [Measured results and paired excerpts](results/live_evaluation_report.md)

### Setup and request budget

From the repository root, activate the virtual environment and create `.env` from
`.env.example` if needed. Add the following entries to the ignored file using your local editor.

```dotenv
GEMINI_API_KEY=<your Gemini API key>
GEMINI_MODEL=gemini-3.5-flash-lite
```

Choose the exact model ID available to your Google project. Google lists December 1, 2026 as the
shutdown date for `gemini-2.5-flash` and says requests to that ID will be routed to Gemini 3.5
Flash-Lite from that date. Use the newer model ID directly for a controlled comparison.
[Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations#gemini-2.5-flash-models)

```bash
set -a
source .env
set +a
python -m evaluation.live_experiments plan --provider gemini
```

After approving external usage, run the smoke test first. A validated Gemini explanation is
required before proceeding with the retrieval and prompt comparisons.

```bash
python -m evaluation.live_experiments smoke --provider gemini --yes
python -m evaluation.live_experiments retrieval --provider gemini --yes
python -m evaluation.live_experiments prompts --provider gemini --yes
```

The standalone assistant and the optional five-case evaluation harness can also select Gemini.
The harness reports validated provider explanations and template fallbacks separately. Its budget
is separate from the three experiments below.

```bash
python -m src.main --ask "Music for studying" --provider gemini
python -m evaluation.run_evaluation --live --provider gemini --yes
```

To use OpenRouter, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` and pass
`--provider openrouter`. OpenRouter requests JSON through the shared prompt and reads
`choices[0].message.content`. Gemini uses the native `generateContent` endpoint with the same
prompt and an explicit JSON response schema. It caps the output at 4,096 tokens. Gemini 3.5
Flash-Lite uses `MINIMAL` thinking and Gemini 2.5 Flash uses a zero thinking budget. Sampling
uses the selected model's default. The application still validates generated content before
using it. [Gemini REST reference](https://ai.google.dev/api/generate-content),
[Structured output](https://ai.google.dev/gemini-api/docs/structured-output),
[Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)

| Command suffix | Initial logical requests | Possible content corrections | Maximum HTTP attempts |
|---|---:|---:|---:|
| `smoke --provider gemini --yes` | 1 | 1 | 3 |
| `retrieval --provider gemini --yes` | 6 | 0 | 12 |
| `prompts --provider gemini --yes` | 6 | 6 | 18 |
| Total | 13 | 7 | 33 |

A logical request can use one transport retry. Smoke and prompt arms share a three-attempt cap
across the initial answer and correction. The planned maximum is 20 logical requests including
corrections. The earlier Phase 3 plan reported 13 logical requests and 26 HTTP attempts because
it captured only first answers. The dry run `python -m evaluation.live_experiments all --mock`
exercised 13 simulated initial calls. Those calls sent zero HTTP requests.

### Evidence and comparison protocol

- **Smoke.** One complete assistant interaction, from request validation to the final explanation.
  Review song IDs, source IDs, accepted generation stage, and the final generator in
  `results/live_gemini_smoke.json` and `traces/live_gemini_smoke.json`. A template fallback fails the smoke
  command even if the assistant returns usable recommendations. The `all` command stops after a
  failed smoke.
- **Retrieval.** In `results/live_gemini_retrieval.json`, compare first answers across three matched
  requests using the specialized prompt. The two arms change only the `sources` list. Both keep
  the request, plan, provenance metadata, authoritative limitations, ranked results, attributes,
  scores, schema and generation settings. The withheld arm may still see source IDs in the fixed
  provenance or limitations. It receives no passage text and has no allowed citation IDs.
  Review relevant contextual claims, use of evidence, supported citations, disclosure and coverage.
  Quote a paired example only after checking both answers against their saved grounding.
- **Prompt specialization.** In `results/live_gemini_prompts.json`, compare general and specialized
  prompts on the same three requests with the same grounding and generation settings. Report
  initial acceptance over all three requests per style, corrected acceptance over requests that
  reached correction, and final accepted responses split by generator. Keep provider errors
  and rejected answers in the denominators for initial success. Template fallback belongs in the
  final workflow count. In the model card, include a paired excerpt and discuss differences in
  musical detail, inferred preference attribution, unmet preferences, catalog limits and clarity.

Results and corresponding files in `traces/` record the requested model, reported model version
and finish reason when available, and HTTP attempts. Results also retain sanitized raw answer text,
errors and automatic metrics. A Gemini response from a different model is rejected. If the served
version varies across comparison arms, the runner saves the evidence and fails the comparison.
The figures check JSON decodability,
guardrail acceptance, song references, explanation coverage and supplied citation IDs. Keyword
counts for provenance, feature mentions and missed preferences, and the presence of a limitations
field, are screening measures. Manually verify factual accuracy, whether the cited passage
supports the claim, relevance to the listening request, and clarity. Keep any evidence of failure
with its original output and the failure reason. Three pairs provide descriptive comparisons.

### HTTPS troubleshooting

The retained OpenRouter smoke reported `CERTIFICATE_VERIFY_FAILED`. The earlier Gemini
diagnostics did not preserve its exception. On the macOS Python 3.14.7
installation used by this project, the default OpenSSL CA file and directory were missing.
With the shell's CA override removed, both providers' public endpoints failed certificate
verification with code 20. Both verified successfully with `/etc/ssl/cert.pem`. The shared
provider transport now uses that macOS bundle when Python has no default CA file or directory
and no explicit CA override. Certificate and hostname checks stay enabled. Python.org also
ships a certificate installer for its macOS distribution. [Python on macOS](https://docs.python.org/3/using/mac.html)

The trace now distinguishes certificate failures, proxy tunnel errors, and timeouts during
connection or request write, response headers, and response body reads. It reports exception
types and known diagnostic codes. Raw upstream exception messages stay out of the trace.
Certificate verification failures stop immediately. Transient failures still use the existing
retry budget. An unauthenticated HTTP 403 from Gemini's models endpoint confirmed HTTPS
connectivity. The subsequent end-to-end Gemini smoke separately verified live generation.
[OpenRouter failure trace](traces/live_smoke.json) · [Successful Gemini trace](traces/live_gemini_smoke.json)

## Reliability review

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
  prompt-injection defenses have limited coverage; live comparisons used one model, three
  synthetic requests, and one answer per request-arm pair.
- **Focused checks:** the harness and tests exercise fabricated song IDs, unsupported citation
  IDs, fabricated decimals, genre and artist constraints, contradictory requests, empty input,
  and prompt injection attempts. The Phase 4 record checks the 500-character boundary. Mock and
  fake HTTP tests cover malformed answers, unavailable providers, retry and correction limits,
  and template fallback. The runner now retains malformed citations when measuring failed
  responses. The prose number guard checks some decimals against supplied values. It has no
  general check for integer claims, units or whether an allowed number describes the right feature.
- **Gemini-specific checks:** mocked requests cover authentication, unavailable models, quota
  exhaustion, rate-limit retries, safety blocks, truncation and credential redaction. A provider
  token-limit stop fails the response and retains the sanitized partial answer in experiment
  evidence. Malformed explanation JSON can trigger the shared one-correction path. A safety-block
  or token-limit stop goes directly to template fallback.
