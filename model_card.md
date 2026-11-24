# VibeFinder AI model card

## System overview

VibeFinder AI is a hybrid recommendation assistant for a fictional song catalog. A rule-based
interpreter extracts preferences and constraints from a natural-language request. A local TF-IDF
index retrieves guidance and catalog facts. The planner records which preferences are explicit,
inferred or unknown, then calls the original deterministic weighted scorer. It separately enforces
hard constraints and can use diversity reranking for softer variety preferences.

The local template is the default explanation provider. When selected, an external Gemini or
OpenRouter model writes structured explanation text from verified rankings, track attributes,
preference provenance and retrieved sources. The engine owns the catalog, scores and order. The
workflow validates the returned explanation, allows at most one content correction with a ceiling
of three HTTP attempts, and falls back to a deterministic template if live generation fails.
[Architecture](docs/architecture.md) · [Original Module 3 model card](docs/archive/module3/model_card.md)

## Intended use

The system is designed for a classroom demonstration of interpretable ranking, local retrieval,
and grounded explanations. A listener can describe a context or a preference and inspect how a
small catalog fits it. A developer or grader can reproduce the behavior with the offline provider,
traces and predefined cases. The score represents weighted fit to active preferences in this catalog.
It should be used as a relative ranking signal for the request.

## Dataset and knowledge sources

- [Song catalog](data/songs.csv) contains 18 fictional songs by 16 fictional artists, across 13
  genres and nine moods. Attributes such as energy, acousticness and popularity are synthetic.
  There are no streamed recordings, play logs or listener profiles backing the catalog.
- The [listening-context guide](data/knowledge/listening_contexts.md) describes contexts such as
  studying, workouts and quiet evenings. Its hints are starting assumptions. The assistant labels
  their provenance and preserves explicit preferences.
- The [music-features guide](data/knowledge/music_features.md) explains attributes and ranking.
  Catalog-derived passages expose genre counts, candidate facts, and coverage limits. The combined
  index has 62 passages, including five from the listening-context guide. [Index comparison](evaluation/results/retrieval_comparison.md)

## Model and specialization

Google Gemini served `gemini-3.5-flash-lite` throughout the measured runs. Both prompt arms
received identical grounding, used the same response schema, and used a 4,096-token cap,
`MINIMAL` thinking and the provider's default temperature. Specialization adds music-specific
instructions and two few-shot examples. **No model weights were trained or fine-tuned.**

### Prompt specialization comparison

Each arm generated one first answer for each of three matched synthetic requests.

| Measure | General prompt | Music-specialized prompt |
|---|---:|---:|
| First answers passing full validation | 3/3 | 3/3 |
| Content corrections | 0 | 0 |
| Final live Gemini explanations | 3/3 | 3/3 |
| Template fallbacks | 0 | 0 |
| Ranked songs explained | 15/15 | 15/15 |
| Items mentioning a numeric track feature | 10/15 | 15/15 |
| Guide or inference wording on applicable items, keyword count | 0/10 | 10/10 |
| Correct structured unmet-preference fields on applicable items | 7/7 | 7/7 |
| Explicit item-level preference-miss disclosure, qualitative review | 0/7 | 7/7 |
| Valid structured citation IDs among emitted IDs | 29/29 | 20/20 |
| Responses with structured item citations | 3/3 | 2/3 |
| Mean words per song explanation | 16.5 | 23.6 |

For the very high-energy acoustic request, these are the actual Gym Hero explanations:

- **General:** “Gym Hero features an energy level of 0.93 compared to your 0.95 target, but its acousticness is 0.05.”
- **Specialized:** “Its very high energy (0.93) is close to your target, but its low acousticness (0.05) misses your acoustic preference.”

Explicit disclosure helps the reader see which part of the request a compromise fails without
interpreting a raw acousticness value. In this sample the specialized style made more such
disclosures and used more guide-derived wording. It also produced longer prose and fewer
structured citations. On the acoustic request the general arm used item citations, while the
specialized arm used empty citation arrays. The two arms retained the catalog limitation.
The provenance figure counts wording. It does not independently test intent recognition or the
accuracy of each attribution. [Paired results](evaluation/results/live_gemini_prompts.json) · [Prompt traces](evaluation/traces/live_gemini_prompts.json) · [Scoring definitions and excerpts](evaluation/results/live_evaluation_report.md#prompt-specialization)

## Evaluation methodology and results

**Offline evaluation.** The project-owned harness passed 22/22 cases and 197/197 checks on
interpretation, provenance, ranking, constraints, validation and recovery. Its expectations were
developed alongside the implementation. The counts measure agreement with specified behavior.
All 417 tests passed in fresh Python 3.12 and 3.14 environments, including the 186 original tests.
The seven original CLI and weight-experiment outputs matched byte for byte. [Evaluation cases](evaluation/cases.json) · [Offline results](evaluation/results/offline_evaluation.json)

**Live evaluation.** The saved batch contains 13 logical requests and 13 HTTP attempts. Twelve
first answers passed full validation. The smoke passed 1/1 and the prompt arms passed 3/3 each.
The retrieval comparison passed 3/3 with passages and 2/3 with passages withheld. The retrieval
arms held the ranked results and all grounding fields other than the passage list fixed. Earlier
offline tests also showed that retrieval could fill enough preferences to answer three context-only
requests and that additional knowledge sources improved certain catalog disclosures.
[Live report](evaluation/results/live_evaluation_report.md) · [Offline retrieval comparison](evaluation/results/retrieval_comparison.md)

## Limitations and bias

- **Catalog coverage.** Ten of 13 genres have one song. Some requested genres are absent. The
  synthetic catalog correlates energy with lower acousticness (Pearson `r ≈ −0.96`), which
  explains why very high-energy acoustic requests force a compromise.
- **Manual ranking choices.** Weights and score thresholds are hand-selected. The 40% minimum is
  a heuristic chosen after comparing 0%, 40% and 60% alternatives. Tolerance checks also affect
  which preferences count as met. These choices have no listener calibration.
- **Interpretation and inferences.** The interpreter understands a vocabulary of phrases and
  rules. It can overlook unfamiliar wording or apply a guide hint that does not fit an individual.
- **Explanation checks.** Validation catches unknown song IDs, disallowed citation IDs, and some
  numeric problems. It cannot reliably verify whether a passage supports a claim or whether a
  valid number describes the correct feature. The live review found ambiguous phrasing that
  blended guide targets with catalog values.
- **Evidence and availability.** The live sample is one model and three synthetic requests, with
  one answer per arm. There was no independent human review or listener satisfaction study.
  Live generation depends on network access, quota, credentials and provider availability.
  [Quality floor experiment](evaluation/results/quality_floor.md) · [Live limitations](evaluation/results/live_evaluation_report.md#measurement-notes-and-limitations)

## Potential misuse and prevention

| Potential misuse | Implemented mitigation and remaining risk |
|---|---|
| Treating a score as objective music quality | The CLI shows the active maximum, score components and missed preferences. It reports fit within the fictional catalog using the active preferences. |
| Presenting generated song names as real recordings | The catalog is labeled fictional. Result validation verifies membership and scores, and explanation validation checks that the items match the engine's picks. |
| Treating language about mood or context as a psychological assessment | Rules extract musical preferences and listening contexts. Guide inferences are labeled, and explicit requests take precedence. The system has no clinical assessment or listener-profiling data. |
| Injecting instructions through a request or retrieved text | Prompts treat those fields as untrusted data. The external model writes explanations only. The program validates its output and keeps engine results authoritative. Adversarial prose can still reach the model. |
| Fabricating recommendations or presenting weak claims with a citation | The validator checks song IDs, citation IDs, selected numeric facts and the schema. One correction or a template fallback limits invalid output. A valid citation ID can accompany an unsupported explanation. |
| Exposing personal text or API credentials | Offline runs keep requests local. Live providers receive the prompt and grounding. Credentials come from the environment, and known key values are redacted in traces and saved evidence. Requests can still contain sensitive text, so the published examples use synthetic prompts. |

## Reliability surprise

Hundreds of mocked and offline tests passed before live HTTPS testing exposed a trust-store
problem. Curl reached both providers. Python with its CA override removed failed certificate
verification against both public endpoints because the macOS Python install had no usable
default CA bundle. The retained OpenRouter smoke confirms the verification failure. The earlier
Gemini log retained a generic network error, so its exact exception is unknown. Both providers
now use the system CA bundle when that macOS default and overrides are absent, with certificate
and hostname verification enabled. The transport records sanitized TLS and timeout details.
Template fallback preserved usable recommendations during the failures. [HTTPS diagnosis](evaluation/README.md#https-troubleshooting) · [OpenRouter fallback](evaluation/traces/live_smoke.json) · [Successful Gemini smoke](evaluation/traces/live_gemini_smoke.json)

The rejected retrieval-withheld study answer had valid JSON and explained all five songs. It
cited the studying guide in every item. That ID remained visible in fixed provenance, while
the allowed source list was empty. The retained ID may have cued the invalid citations.
One sample per arm leaves the effect on validation rates uncertain.
[Rejected answer and analysis](evaluation/results/live_evaluation_report.md#retrieval-and-generation)

## AI collaboration reflection

### An AI suggestion I accepted

One suggestion I accepted was introducing `NO_PREFERENCE` into the recommendation engine. In my
original project, users had to provide every required preference before receiving recommendations.
This became a problem when I introduced natural-language requests because users do not always know
or specify exactly what they want. This suggestion allowed me to handle missing preferences without
assigning values the user never requested or breaking the original recommendation system.
[Phase 2 interaction record](ai_interactions.md#phase-2--core-recommendation-assistant)

### An AI suggestion I rejected or revised

The AI initially suggested using Anthropic as the external model provider and creating four
separate knowledge guides. I decided to make the providers configurable and reduce the knowledge
base to two guides supported by information from the song catalog. I eventually chose Gemini
because I already had access to its API. I wanted to keep the system flexible and avoid adding
complexity that was unnecessary for the project.
[Design audit](ai_interactions.md#phase-1--repository-audit-and-system-design) · [Phase 4 record](ai_interactions.md#phase-4--gemini-integration-debugging-and-live-experiments)

### Why those decisions mattered

These decisions helped me build on my original project instead of replacing it entirely. They
also made the system easier to test, maintain, and run without requiring an API key. Having
multiple provider options and an offline fallback meant that the application could still provide
recommendations even when an external AI service failed.
[Multi-source comparison](evaluation/results/retrieval_comparison.md) · [OpenRouter fallback trace](evaluation/traces/live_smoke.json)

### What I learned

One of my biggest lessons was that passing automated tests does not necessarily mean a system
will work in a real environment. Both Gemini and OpenRouter failed despite hundreds of passing
tests. After testing the connections separately, I discovered that the reproduced connection
failure was caused by Python's SSL certificate configuration rather than the recommendation
system itself. [HTTPS diagnosis](evaluation/README.md#https-troubleshooting)

I also learned that a valid AI response is not always an accurate or clear response. Although
most of the Gemini outputs passed validation, reviewing their explanations revealed some
ambiguity between inferred preferences and actual song attributes. This showed me why automated
validation and human review are both important when building AI systems.
[Live evaluation report](evaluation/results/live_evaluation_report.md#retrieval-and-generation)

### What I would improve

If I continued working on this project, I would expand the song catalog, improve how the
assistant understands different ways of expressing music preferences, and evaluate the
recommendations with actual listeners.

I would also conduct more live experiments with repeated requests and independent reviewers.
My current results show promising differences between general and specialized prompting, but
the sample size is too small to draw broader conclusions about their effectiveness.
[Prompt comparison](evaluation/results/live_evaluation_report.md#prompt-specialization)
