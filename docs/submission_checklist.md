# VibeFinder AI submission checklist

This audit maps the CodePath AI110 criteria to repository evidence. Status describes the
available deliverables. The grader determines the awarded points.

## Required criteria — 21 points available

| Criterion | Status | Implementation and evidence | Reproduction | Remaining limitation or risk |
|---|---|---|---|---|
| Original project and scope (3) | Completed | The [README](../README.md#original-project) identifies the Module 3 simulation, links the original repository, and explains what was added. The [Module 3 archive](archive/module3/README.md) preserves the original features and experiments. | `python -m src.main` | The original catalog is intentionally small and fictional. |
| Substantial integrated AI feature (3) | Completed | [Assistant workflow](../src/assistant/workflow.py) integrates intent interpretation, multi-source retrieval, provenance, ranking and verified explanations. The original engine remains the source of scores. | `python -m src.main --ask "Music for studying." --show-plan` | Vocabulary coverage depends on deterministic rules. |
| Mermaid architecture (3) | Completed | The separate [Mermaid source](../diagrams/architecture.mmd) shows current components and branches. The [architecture page](architecture.md) embeds the source and links the [rendered SVG](../assets/architecture.svg). | `mmdc -i diagrams/architecture.mmd -o /tmp/vibefinder-architecture.svg` using Mermaid CLI. | Zoom into sections when using the full diagram on a slide. |
| Functional end-to-end demonstration (3) | Completed | [README examples](../README.md#example-usage) contain captured study, workout and no-match output, plus a verbatim [live smoke](../evaluation/results/live_gemini_smoke.json) excerpt. | Run the two [presentation demo commands](presentation.md#slide-4--offline-terminal-demo-130). | Live quota is unnecessary for the demo. Use the saved output when presenting Gemini. |
| Reliability or guardrails (3) | Completed | [Validation](../src/assistant/validation.py) checks constraints, engine facts and explanation fields. The [no-match trace](../evaluation/traces/offline_no_match_only_metal.json), [simulated fallback](../evaluation/traces/simulated_invalid_then_fallback.json) and [real TLS failure](../evaluation/traces/live_smoke.json) show distinct failure paths. | `python -m src.main --ask "Only recommend metal songs."` and `python -m pytest -q tests/test_agentic.py tests/test_transport.py` | An allowed citation or number can still appear in ambiguous wording. |
| README and reproducibility (3) | Completed | [Setup](../README.md#getting-started) starts from a clone. The default template, [requirements](../requirements.txt) and [optional environment template](../.env.example) separate normal grading from paid providers. | Follow setup, then `python -m pytest -q` and `python -m evaluation.run_evaluation`. | Full runs were verified on macOS with Python 3.12.7 and 3.14.7. |
| Responsible-AI reflection (3) | Completed | The [Model Card](../model_card.md) covers intended use, fictional data, bias, misuse and HTTPS recovery. The author reviewed and approved the [first-person reflection](../model_card.md#ai-collaboration-reflection), which links to the development evidence. | Review the Model Card against the [interaction log](../ai_interactions.md). | The live sample is small and lacks independent reviewer or listener feedback. |

## Stretch criteria — 8 points available

| Criterion | Status | Implementation and evidence | Reproduction | Remaining limitation or risk |
|---|---|---|---|---|
| Multi-source RAG with before/after evidence (+2) | Completed | [Offline before/after](../evaluation/results/retrieval_comparison.md) compares retrieval on/off and a single guide versus the full index. [Live passage comparison](../evaluation/results/live_evaluation_report.md#retrieval-and-generation) holds the songs and plan fixed and reports 3/3 supplied versus 2/3 withheld. | `python -m evaluation.retrieval_comparison`. For live reproduction use the [budgeted provider commands](../evaluation/README.md#setup-and-request-budget) after approval. | The withheld arm retained provenance IDs. One answer per pair leaves the effect on future validation rates uncertain. |
| Agentic workflow with saved traces (+2) | Completed | The [trace index](../evaluation/README.md#traces) includes offline branches, scripted invalid explanations and correction, safe fallback, and successful live generation. `Assistant.ask()` bounds correction and HTTP attempts. | `python -m src.main --ask "Music for studying." --trace-file /tmp/vibefinder-study.json` and `python -m evaluation.generate_traces --check` | Content correction is demonstrated by simulation. The final live batch required none. |
| Measured model specialization (+2) | Completed | The [Model Card comparison](../model_card.md#prompt-specialization-comparison) uses matched inputs on the same Gemini model, two prompt styles, measured trade-offs and actual excerpts. | Review [paired JSON](../evaluation/results/live_gemini_prompts.json) and the scoring definitions. Live reproduction uses the approved `prompts` command in the evaluation guide. | Three request pairs and one model. The prose audit had no independent human reviewer. |
| Automated evaluation on predefined inputs (+2) | Completed | [Harness](../evaluation/run_evaluation.py), [22 fixed cases](../evaluation/cases.json) and [offline results](../evaluation/results/offline_evaluation.json) cover 197 checks. Three scripted cases exercise provider recovery. | `python -m evaluation.run_evaluation` | Expectations were developed alongside the implementation. This is a behavioral test suite. |

## Final verification

- Fresh local clones installed `requirements.txt` in separate Python 3.12.7 and 3.14.7 virtual
  environments. Each passed all 417 tests without provider credentials, including 186 original
  tests. All seven captured CLI and weight-experiment outputs matched byte for byte.
- The default original and assistant CLIs, all presentation commands, offline harness, a newly
  written trace, and offline retrieval comparison ran successfully without credentials. The
  harness passed 22/22 cases and 197/197 checks. The 18-song CSV matches the Module 3 baseline.
- Mermaid CLI rendered the standalone source successfully. The fenced source on the architecture
  page matches the standalone file. Local Markdown links and their referenced headings were
  checked. `git diff --check` passed.
- The saved live artifacts were checked for configured API-key values, authorization headers and
  known credential formats. `.env.example` is tracked as a blank template. Local credential and
  cache files are ignored. Phase 5 made zero additional provider calls.

## Manual submission

1. Rehearse the [6:15 presentation](presentation.md) using its offline terminal commands.
2. From the repository root, inspect and stage the final documentation. Review the staged content
   and verify it contains the Mermaid source and the intended evidence links.

   ```bash
   git status --short
   git add README.md model_card.md ai_interactions.md evaluation/README.md docs/applied_ai_design.md \
     docs/architecture.md docs/archive/module3/README.md docs/archive/module3/model_card.md \
     docs/portfolio.md docs/presentation.md docs/submission_checklist.md \
     diagrams/architecture.mmd assets/architecture.svg
   git diff --cached --check
   git diff --cached --stat
   git diff --cached
   ```

3. Commit and push the reviewed files when ready.

   ```bash
   git commit -m "docs: finalize VibeFinder AI submission"
   git push origin main
   ```

4. Open the [repository](https://github.com/GabrielAnokye/vibefinder-ai) on GitHub. Verify that
   the README, Model Card, Mermaid source and saved evidence are accessible to the grader.
   Paste the repository URL into the CodePath AI110 final-project assignment. Follow any
   additional attachment or recording instructions on that form.
