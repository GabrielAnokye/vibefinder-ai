# VibeFinder AI live evaluation

The project author ran the three Gemini experiments on November 23, 2026. The saved evidence confirms **13 logical requests, 13 HTTP attempts, and 12 initial answers accepted by the full validator**. All 13 answers were JSON objects and finished with `STOP`. The smoke and prompt workflows returned seven validated Gemini explanations. The retrieval experiment scored six first answers directly. There were zero transport retries, corrections, or template fallbacks in this batch. [Commands and budget](../README.md#live-provider-evaluation)

| Experiment | Initial answers | Accepted | Results | Trace |
|---|---:|---:|---|---|
| End-to-end smoke | 1 | 1 | [JSON](live_gemini_smoke.json) | [Trace](../traces/live_gemini_smoke.json) |
| Passages supplied or withheld | 6 | 5 | [JSON](live_gemini_retrieval.json) | [Traces](../traces/live_gemini_retrieval.json) |
| General or specialized prompt | 6 | 6 | [JSON](live_gemini_prompts.json) | [Traces](../traces/live_gemini_prompts.json) |

## Design and controls

The requested model and every provider-reported version were `gemini-3.5-flash-lite`. All runs used the same JSON response schema, 4,096-token cap, `MINIMAL` thinking, and Gemini's default temperature. Each comparison used one answer per arm on each of three synthetic requests covering studying, a workout with a soft genre preference, and very high-energy acoustic music.

The saved retrieval payloads match field for field except for `sources`, which was emptied in the withheld arm. The specialized prompt ran in both retrieval arms. The saved general and specialized payloads are identical. The offline workflow reproduced the fixed plan, five ranked songs, attributes, scores, and limitations for every request. Retrieval-derived provenance and limitations stayed in the withheld payload. This measures the incremental effect of passage text on explanation generation. [Experiment runner](../live_experiments.py)

## End-to-end smoke

`Music for studying.` returned Focus Flow, Midnight Coding, Library Rain, Coffee Shop Stories, and Spacewalk Thoughts in engine order. Gemini explained all five songs and passed validation on its first HTTP attempt. Its six structured citations used `listening_contexts#studying` for all five songs and `music_features#vocal-intensity` once. The instrumentalness passage was supplied and received no structured citation. This is the actual explanation for Focus Flow, song 9. [Smoke output](live_gemini_smoke.json)

> This track features low energy (0.4) and low vocal intensity (0.03) with high instrumentalness (0.9), and matches the focused mood tag suggested by the studying guide.

On the same input, the offline template stated the engine score and listed matched inferred targets. Gemini gave the actual feature values in prose and averaged 23.0 words per item versus 35.0 for the template. The feature values and tags in the smoke answer agree with the grounding. One wording concern is Library Rain's “Suggested by the studying guide, this song has …” opening, which can read as a track endorsement. The guide provides listening heuristics and the engine selects the track.

## Retrieval and generation

| Initial-answer measure | Passages supplied | Passages withheld |
|---|---:|---:|
| Valid JSON | 3/3 | 3/3 |
| Full application validation | 3/3 | 2/3 |
| Valid song references and ranked-song coverage | 15/15 each | 15/15 each |
| Valid structured citation IDs among emitted citations | 25/25 | 0/5 on the study answer, N/A for the other two |
| Items with a musical feature keyword | 15/15 | 15/15 |
| Items with an actual numeric feature value | 15/15 | 15/15 |
| Items using guide or inference keywords where they fit inferred targets | 7/10 | 8/10 |
| Correct structured unmet-preference fields on applicable items | 7/7 | 7/7 |
| Unmet preference explicitly stated in item prose, qualitative review | 7/7 | 7/7 |
| Applicable response retaining both catalog limitations | 1/1 | 1/1 |
| Mean words per item | 21.1 | 23.1 |

**Studying, Focus Flow (song 9).** The supplied answer attached citations for the study context and two feature definitions. The withheld answer still had the track values and the guide's ID in provenance.

- **Passages supplied:** “This track fits the studying guide's suggested energy (0.4), high instrumentalness (0.9), and low vocal intensity (0.03).”
- **Passages withheld:** “Focus Flow is a lofi track with energy 0.4, instrumentalness 0.9, vocal intensity 0.03, and mood tags including focused. These targets were suggested by the studying guide.”

The withheld study answer was rejected solely for five instances of `cites unsupported source(s): listening_contexts#studying`, one each for song IDs 9, 2, 4, 7, and 6. The ID was visible in `inferred_sources` and absent from the empty allowed `sources` list. Copying it into the structured citation field violated the prompt's citation rule. The retained provenance is a plausible contributing cue. With one draw per arm, the study pair cannot establish whether withholding passages increased failure risk.

**Workout, Breakwater (song 16).** The supplied answer cited the workout guide, which explicitly connects movement with high energy and a produced sound. Both items could explain the musical fit from the fixed attributes and provenance.

- **Passages supplied:** “This rock song has high energy (0.87) and low acousticness (0.07), matching your workout request and the workout guide's suggested non-acoustic style.”
- **Passages withheld:** “An intense rock track with high energy (0.87) that matches your genre and energy preferences. Less acoustic sound (0.07) and energetic mood tags were suggested by the guide.”

**Acoustic music with very high energy, summary.** The catalog passage supplies context about the trade-off. It reports that the seven tracks with acousticness at least 0.5 reach at most 0.42 energy. Both payloads already contained the engine's limitation, and both answers preserved it. Only the supplied answer brought the trade-off into its opening summary.

- **Passages supplied:** “The catalog does not contain any high-energy acoustic songs, so these results represent the closest available compromises.”
- **Passages withheld:** “You are looking for acoustic music with very high energy.”

**Qualitative grounding review.** The track feature values matched the catalog in both arms. All five supplied study items used wording that could blur a guide target and a track value. For example, Focus Flow's song energy is 0.40 and the guide's target is 0.35. The withheld Library Rain item said the guide suggested “these specific feature ranges,” although the stored plan gives target values. Both workout answers identified the two pop substitutions. The supplied answer cited the workout passage for the high-energy, non-acoustic rationale, while the withheld answer described a similar rationale from the fixed plan. The supplied acoustic answer cited the feature definition for each song. The withheld acoustic answer repeated a bracketed catalog ID copied from the fixed limitation. The formal citation metric checks structured item citations and leaves that inline ID outside its denominator.

The earlier **offline planning** comparison answered all three context-only requests when retrieval was enabled and asked for clarification with retrieval off. The **offline multi-source** comparison measured 62 available passages versus five from the contexts guide alone. The larger index enabled labeled substitutes for a soft metal request and provided a catalog citation for the high-energy acoustic limitation. The **live generation** experiment shows a change in how explanations cite evidence and foreground limitations. The recommendations themselves were fixed. [Offline retrieval experiments](retrieval_comparison.md)

## Prompt specialization

| Measure | General | Music-specialized |
|---|---:|---:|
| Valid JSON and full initial validation | 3/3 each | 3/3 each |
| Corrected responses attempted | 0 | 0 |
| Final Gemini explanations, template fallbacks | 3/3, 0 | 3/3, 0 |
| Valid song references and complete ranked-song coverage | 15/15 each | 15/15 each |
| Items with a musical feature keyword | 15/15 | 15/15 |
| Items with an actual numeric feature value | 10/15 | 15/15 |
| Guide or inference keywords where items fit inferred targets | 0/10 | 10/10 |
| Valid structured citation IDs among emitted citations | 29/29 | 20/20, N/A on the acoustic answer |
| Answers with structured item citations | 3/3 | 2/3 |
| Correct structured unmet-preference fields on applicable items | 7/7 | 7/7 |
| Unmet preference explicitly stated in item prose, qualitative review | 0/7 | 7/7 |
| Applicable response retaining both catalog limitations | 1/1 | 1/1 |
| Mean words per item | 16.5 | 23.6 |

**Studying, Focus Flow (song 9).** General states the musical fit concisely. Specialized adds catalog values and names the source of the inferred mood tag.

- **General:** “Focus Flow features low vocal intensity, high instrumentalness, and the focused mood tag, fitting study sessions well.”
- **Specialized:** “This lofi track has an energy of 0.4, an instrumentalness of 0.9, a low vocal intensity of 0.03, and features the focused mood tag suggested by the studying guide (listening_contexts#studying).”

**Very high-energy acoustic music, Gym Hero (song 5).** General gives the explicit energy target and a catalog citation. Specialized makes the acoustic miss explicit and leaves the item citation array empty.

- **General:** “Gym Hero features an energy level of 0.93 compared to your 0.95 target, but its acousticness is 0.05.”
- **Specialized:** “Its very high energy (0.93) is close to your target, but its low acousticness (0.05) misses your acoustic preference.”

**Qualitative grounding and clarity.** General discloses pop substitutions at the response level and gives Concrete Crown a track-specific catalog citation. Specialized discloses both genre misses next to the songs and cites the workout guide for its contextual advice. General's “steady energy” for Coffee Shop Stories goes beyond a single catalog energy estimate. Specialized's “suggested by the workout guide” next to a pop track can imply the guide recommended the track or genre. Specialization improved item-level attribution and disclosure in this sample, with longer prose and fewer structured citations. Both styles kept complete recommendation coverage and the applicable catalog limitation. This experiment changes prompts and few-shot examples.

## Measurement notes and limitations

Counts pool the saved initial answers, including the rejected study answer. Each arm has 15 song items. Inference has ten applicable items across studying and workout. Unmet preferences have seven across workout and acoustic requests. The one applicable engine-level limitation response is the acoustic request. Citation validity divides allowed structured IDs by emitted structured IDs, including repeated citations across items. A response with zero emitted citations has no per-response validity denominator. Numeric coverage counts items with at least one decimal matching a track feature to two decimal places. Mean length counts whitespace-separated words in the `explanation` fields. The explicit-disclosure review requires an item to name the miss or state it by contrasting with the user's requested sound or genre. Numeric detail, implied misses, and response-level summaries alone do not meet that definition.

The source and keyword checks screen for consistency and wording. They leave claim-to-citation support, perceived clarity, and musical satisfaction for qualitative or listener evaluation. Only one model and three request pairs were sampled. There was no independent human reviewer or listener study, and no estimate of run-to-run variability.

## HTTPS failure and recovery

The earlier Gemini and OpenRouter generation attempts fell back to deterministic explanations. The retained OpenRouter run recorded a certificate verification failure. The earlier Gemini diagnostics were too generic to establish its exact exception. Curl reached both providers, and a controlled credential-free Python probe with no CA override reproduced verification code 20 against both. The macOS Python installation had no usable default CA bundle. The shared transport now uses the system bundle when Python has no configured CA source, keeps certificate and hostname verification enabled, and records sanitized connection and timeout diagnostics. Both public endpoints passed verified HTTPS after the fix, and the subsequent Gemini smoke returned validated live prose. During the outage, the engine could still return recommendations and template explanations. [Retained OpenRouter result](live_smoke.json) · [Retained OpenRouter trace](../traces/live_smoke.json) · [HTTPS diagnosis](../README.md#https-troubleshooting)

The Phase 4 completion check ran without provider credentials. All 417 tests, 22 offline cases,
and 197 offline checks passed. All seven Module 3 golden outputs matched byte for byte, and the
original catalog matched its baseline. The live JSON results and traces passed the check for
configured key values, authorization headers, and known credential patterns.
