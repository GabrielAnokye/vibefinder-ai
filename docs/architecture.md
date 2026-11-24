# VibeFinder AI architecture

Blue nodes run deterministic code. Violet nodes call an optional external LLM. Green nodes
provide local data and guides. The dashed edges connect the evaluation and tracing tools.

```mermaid
%%{init: {"theme": "neutral", "flowchart": {"htmlLabels": false, "curve": "linear", "nodeSpacing": 18, "rankSpacing": 22}}}%%
flowchart LR
    subgraph rankFlow["Deterministic planning and ranking"]
        direction TB
        Eval["Offline evaluation harness"] -. Predefined requests .-> Check
        User([Natural-language request]) --> Check{"Input passes validation?"}
        Check -->|Invalid| InputError([Input error])
        Check -->|Valid| Intent["Rule-based intent interpretation"]
        Intent --> Retrieve["Multi-source TF-IDF retrieval"]
        Knowledge["Listening-context guide<br/>Music-features guide<br/>Catalog-derived facts"] --> Retrieve
        Retrieve --> Plan["Recommendation planner<br/>Provenance: explicit, inferred, unknown"]
        Plan -->|Ambiguous or insufficient| Clarify([Ask for clarification])
        Plan -->|Impossible hard requirement| NoMatch([No matching songs])
        Plan -->|Usable plan| Filter["Hard constraint filtering"]
        Catalog[("18-song catalog")] --> Filter
        Filter -->|No candidates| NoMatch
        Filter -->|Candidates| Rank["Original weighted engine"]
        Rank --> Coverage["Coverage and limitations<br/>Artist uniqueness and quality floor"]
        Coverage -->|No eligible results| NoMatch
        Coverage -->|Eligible results| Verify{"Results valid?"}
        Verify -->|Rejected| ResultError([Result error])
        Verify -->|Accepted| Ground["Verified rankings and grounding"]
    end

    subgraph explainFlow["Explanation workflow · Max 3 HTTP attempts"]
        direction TB
        Select["Select explanation provider"]
        Select -->|Offline or unconfigured| Template["Offline template provider"]
        Select -->|Gemini| Gemini["Gemini"]
        Select -->|OpenRouter| Router["OpenRouter"]
        Gemini --> Answer{"Initial explanation valid?"}
        Router --> Answer
        Answer -->|Invalid with budget left| Correct["At most one content correction"]
        Correct --> Recheck{"Corrected explanation valid?"}
        Answer -->|Failure or no budget| Fallback["Safe template fallback"]
        Recheck -->|Invalid or provider fails| Fallback
        Fallback --> TemplateCheck["Validate template"]
        Template --> TemplateCheck
        TemplateCheck --> Output(["Final recommendations<br/>Scores, limits and explanation source"])
        Answer -->|Valid| Output
        Recheck -->|Valid correction| Output
        Output -.-> Trace["Observable execution traces<br/>All outcomes, decisions and attempts"]
    end

    rankFlow -->|Verified payload only| explainFlow

    classDef llm fill:#f3e8ff,stroke:#7041a0,color:#2c153d,stroke-width:2px
    classDef deterministic fill:#e8f2fb,stroke:#346082,color:#142d42
    classDef data fill:#e9f5ec,stroke:#408056,color:#183a26
    classDef evidence fill:#fff4db,stroke:#93702b,color:#43310f
    classDef terminal fill:#f2f4f6,stroke:#657481,color:#202b35
    class Gemini,Router llm
    class Check,Intent,Retrieve,Plan,Filter,Rank,Coverage,Verify,Ground,Select,Template,Answer,Correct,Recheck,Fallback,TemplateCheck deterministic
    class Knowledge,Catalog data
    class Eval,Trace evidence
    class User,InputError,ResultError,Clarify,NoMatch,Output terminal
```

The planner labels explicitly stated preferences, eligible guide inferences, and unknowns.
The original engine owns the scores and rankings. The explanation workflow receives only
verified results plus the grounding used to explain them. The one content correction shares
a maximum of three HTTP attempts with the initial call and any transport retry. The offline
template can explain the same results if a live provider fails.

The evaluation harness drives the assistant with fixed requests. Traces record observable
stages, decisions, validation results, provider outcomes and attempt counts.

[Mermaid source](../diagrams/architecture.mmd) · [Rendered diagram](../assets/architecture.svg) · [Evaluation and traces](../evaluation/README.md)
