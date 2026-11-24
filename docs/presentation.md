# VibeFinder AI presentation outline

Planned duration is **6 minutes 15 seconds**, including the offline terminal demo. Before
presenting, finish the [README setup](../README.md#getting-started), activate the virtual
environment, and open the terminal at the repository root. Preopen the diagram and the live
report so the demo and evidence can fit inside the same time limit.

## Slide 1 — VibeFinder AI and Module 3 (0:35)

**On slide**

- VibeFinder AI · CodePath AI110 Applied AI System
- Built on the Module 3 weighted music recommender
- An 18-song fictional catalog with explainable scores

**Speaker notes.** “I started from my Module 3 recommender. It takes a structured profile and
ranks a small fictional catalog. I kept that scoring engine and built an assistant around it,
so the way someone asks for music can be more natural.”

**Visual.** Show the title beside an excerpt of the original ranking output. [Original CLI output](../tests/golden/main_default.txt)

## Slide 2 — A request is more natural than a profile (0:40)

**On slide**

- “Music for studying.”
- No stated genre, mood or numeric energy target
- Guide-derived targets are visible and explicit preferences take precedence

**Speaker notes.** “A listener can describe a situation without knowing the available fields.
The study guide can suggest lower vocal intensity and energy. Every suggestion is labeled with
its source. An unknown preference contributes zero points, so the scorer does not need to
invent a genre.”

**Visual.** Highlight two of the inferred fields in the [study example](../README.md#studying).

## Slide 3 — Architecture and ownership (0:55)

**On slide**

- Interpret → retrieve → plan → rank → validate → explain
- Weighted engine owns songs, scores and ranking
- Local templates by default, optional Gemini and OpenRouter

**Speaker notes.** “Blue is deterministic code and violet is external generation. Retrieval
searches the two guides and catalog facts. The planner filters hard requirements before
ranking. The final stage passes verified results and retrieved text to an explanation provider.
I can inspect what happened using the trace.”

**Visual.** Start with the whole diagram, then zoom into the planning and explanation sections
in turn. [Architecture SVG](../assets/architecture.svg) · [Mermaid page](architecture.md)

## Slide 4 — Offline terminal demo (1:30)

**On slide**

- Study request → inferred preferences and ranked results
- “Only metal” → explicit catalog no-match

**Run these commands in order.**

```bash
python -m src.main --ask "Music for studying."
python -m src.main --ask "Only recommend metal songs."
```

**Speaker notes.** “Look at the study guide source and the score reasons for Focus Flow.
The explanation is a local template. For the second request, ‘only’ is a hard constraint.
The catalog cannot satisfy it, so the assistant reports a no-match and offers no substitute.”

**Visual.** Keep the terminal readable at presentation zoom. Highlight the study provenance,
the first pick, and the no-match line. If the terminal is unavailable, show the saved stdout.
[Captured offline demos](../evaluation/results/demo_outputs.md)

Use this read-only command if showing a live explanation without internet or available quota.
It reads the successful Gemini smoke artifact locally. Show the saved excerpt during slide 6
or substitute it for part of the terminal narration to stay within the timing.

```bash
python - <<'PY'
import json
from pathlib import Path

run = json.loads(Path("evaluation/results/live_gemini_smoke.json").read_text())["runs"][0]
final = run["final"]
print(final["generator"])
print(final["explanation"]["items"]["9"])
PY
```

## Slide 5 — Reliability and observable decisions (0:50)

**On slide**

- Validate song and citation IDs
- At most one content correction and three HTTP attempts
- Template fallback preserved the result during a real HTTPS failure

**Speaker notes.** “The validator caught a disallowed citation in the scripted trace. The
workflow requested one correction. The live checks exposed a failure outside the mocked
transport. The macOS Python installation lacked a usable default CA bundle. The
transport now loads a verified system bundle in that case. The fallback kept the recommendations
available while generation was failing.”

**Visual.** Show only the rejection and correction events. Label that trace **simulated**.
Then point to the separate real OpenRouter fallback and validated Gemini smoke.
[Simulated correction](../evaluation/traces/simulated_invalid_then_corrected.json) · [OpenRouter fallback](../evaluation/traces/live_smoke.json) · [Live Gemini smoke](../evaluation/traces/live_gemini_smoke.json)

## Slide 6 — Evaluation, retrieval and specialization (1:00)

**On slide**

- 417 tests · 22/22 offline cases · 197/197 checks
- Gemini accepted 12/13 first answers
- Passage comparison 3/3 supplied, 2/3 withheld
- Both prompt styles passed 3/3. Specialized stated more preference misses and used longer prose.

**Speaker notes.** “The retrieval experiment kept the songs, scores and plan fixed. The
withheld study answer tried to cite an ID it could still see in provenance, even though the
list of permitted citations was empty. That explains the rejection for this one answer.
In the prompt comparison, the specialized prompt explicitly named seven of seven applicable
misses in item prose. The general prompt named none under the same rule, though it copied the
structured fields correctly. Specialized prose averaged 23.6 words per item versus 16.5.”

**Visual.** Show the two comparison tables and, if time allows, the saved Gym Hero pair.
[Live report](../evaluation/results/live_evaluation_report.md) · [Model Card comparison](../model_card.md#prompt-specialization-comparison)

## Slide 7 — Lessons, limits and next evaluation (0:45)

**On slide**

- Regression tests, live connectivity and prose quality answer different questions
- A fictional catalog and three request pairs limit conclusions
- Next evaluation could use repeated trials and independent claim review

**Speaker notes.** “The original output still matches byte for byte. The evaluation showed
that the assistant can explain verified song facts, but that a valid citation can still
accompany ambiguous wording. I would test more varied requests and ask an independent
reviewer to assess whether each cited source actually supports the explanation.”

**Visual.** Use the Model Card’s reliability section and retain the synthetic-catalog label.
[Reliability and limitations](../model_card.md#reliability-surprise)
