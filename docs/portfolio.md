# VibeFinder AI — Intelligent Music Recommendation Assistant

[Public repository](https://github.com/GabrielAnokye/vibefinder-ai)

I extended my CodePath Module 3 music recommender so a listener can ask for “music for studying” without knowing which scoring fields to fill out. VibeFinder AI interprets the request, looks up guidance in a local knowledge index, and uses the original weighted engine to rank a fictional 18-song catalog. It shows which preferences came from the listener, which came from the guides, and where the catalog falls short. An optional Gemini or OpenRouter model explains the verified results. The default template works offline, and the same template can take over if a provider fails. I evaluated the system with regression tests, predefined scenarios, and live Gemini comparisons that held the ranked songs fixed. The specialized prompt made preference misses more explicit in the small live sample.

## Technologies

Python standard library, CSV, local TF-IDF retrieval, pytest, Mermaid, Gemini and OpenRouter APIs.

## Main features

- Natural-language requests with explicit, inferred and unknown preferences.
- Weighted rankings, hard constraints, diversity reranking and catalog limitations.
- Grounded explanations with one possible correction, offline fallback and JSON execution traces.

## Verified results

- 417 tests passed, including 186 original tests. Seven original outputs matched byte for byte.
- The offline evaluation passed 22/22 cases and 197/197 checks. The assistant accepted 12/13 first Gemini answers in the live batch.
- Both prompt styles passed 3/3 requests. The specialized style explicitly disclosed 7/7 applicable preference misses in item prose versus 0/7 in the general style, under the documented review definition. [Evaluation evidence](../evaluation/results/live_evaluation_report.md)

## Optional résumé bullet

- Extended a Python music recommender with natural-language interpretation, local retrieval, and validated LLM explanations, preserving 186 legacy tests and adding a reproducible 22-case evaluation.
