"""
Typed data structures shared by the assistant modules. No logic beyond small helpers.

Provenance of every preference:
    explicit  - stated by the user (the matched phrase is kept as the source)
    inferred  - filled from a retrieved knowledge section (its source id is kept)
    unknown   - not stated and not safely inferable; sent to the engine as NO_PREFERENCE
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EXPLICIT = "explicit"
INFERRED = "inferred"
UNKNOWN = "unknown"

DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10        # safe maximum per request (the catalog has 18 songs)
MAX_REQUEST_CHARS = 500

# Preference fields the interpreter and planner understand, in display order.
PREFERENCE_FIELDS = (
    "genres", "moods", "energy", "acoustic", "popularity", "decade",
    "instrumentalness", "vocal_intensity", "mood_tags",
)
ADVANCED_FIELDS = ("popularity", "decade", "instrumentalness", "vocal_intensity", "mood_tags")

# Listening contexts the interpreter recognizes; each has a section in listening_contexts.md.
CONTEXTS = ("studying", "workout", "relaxing", "nostalgia", "discovery")


@dataclass
class Pref:
    """One preference value with where it came from."""
    value: Any = None
    provenance: str = UNKNOWN
    source: str = ""          # user phrase (explicit) or source id (inferred)

    @property
    def known(self) -> bool:
        return self.provenance != UNKNOWN


@dataclass
class Interpretation:
    """What the rule-based interpreter understood from the request text."""
    text: str
    contexts: List[Tuple[str, str]] = field(default_factory=list)   # (context, phrase), in text order
    prefs: Dict[str, Pref] = field(default_factory=lambda: {name: Pref() for name in PREFERENCE_FIELDS})
    genre_requirement: str = "none"      # "none" | "soft" | "hard"
    unavailable_genres: List[str] = field(default_factory=list)
    excluded_genres: List[str] = field(default_factory=list)
    excluded_artists: List[str] = field(default_factory=list)
    unique_artists: bool = False
    diversity: Pref = field(default_factory=Pref)
    strategy: Pref = field(default_factory=Pref)
    count: Pref = field(default_factory=lambda: Pref(DEFAULT_RESULT_COUNT, UNKNOWN, "default"))
    ambiguities: List[str] = field(default_factory=list)
    clarification: Optional[str] = None

    @property
    def context_names(self) -> List[str]:
        return [name for name, _ in self.contexts]


@dataclass(frozen=True)
class Passage:
    """A retrievable piece of knowledge with a stable source id."""
    source_id: str
    title: str
    text: str
    origin: str                                   # "guide" or "catalog"
    hints: Tuple[Tuple[str, Any], ...] = ()       # validated (key, value) pairs
    applies_when: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    rejected_hints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float


@dataclass
class AppliedHint:
    field: str
    value: Any
    source_id: str
    used: bool                 # False when an explicit user value overrode it
    note: str = ""


@dataclass
class Plan:
    """Executable plan for one request; engine_prefs is passed unchanged to the engine."""
    engine_prefs: Dict[str, Any]
    mode: str
    mode_reason: str
    advanced: bool
    diverse: bool
    diverse_reason: str
    unique_artists: bool
    k: int
    hard_genres: List[str]
    excluded_genres: List[str]
    excluded_artists: List[str]
    hints: List[AppliedHint]
    unknown_fields: List[str]
    limitations: List[str]
    citations: List[str]               # source ids the response may cite
    status: str = "ok"                 # "ok" | "clarification" | "no_match"
    message: Optional[str] = None      # clarification question or no-match explanation


@dataclass
class Recommendation:
    rank: int
    song: Dict[str, Any]
    score: float                       # engine relevance score (never adjusted)
    reasons: List[str]                 # engine reasons, verbatim
    active_max: float
    unmet: List[str]                   # stated preferences this song misses
    adjusted_score: Optional[float] = None
    diversity_note: Optional[str] = None
    pick: Any = None                   # the engine's DiversePick, when diversity reranking ran

    @property
    def share_of_active_max(self) -> float:
        return self.score / self.active_max if self.active_max else 0.0

    @property
    def match_label(self) -> str:
        """Heuristic label, not a probability. 'strong' needs a high score AND no missed preference."""
        share = self.share_of_active_max
        if share >= 0.75 and not self.unmet:
            return "strong"
        if share >= 0.5:
            return "partial"
        return "weak"


@dataclass
class Explanation:
    summary: str
    items: Dict[int, str]              # song id -> prose
    citations: Dict[int, List[str]]    # song id -> source ids
    limitations: List[str]


@dataclass
class AssistantResponse:
    status: str                        # "ok" | "clarification" | "no_match" | "error"
    request: str
    interpretation: Optional[Interpretation] = None
    retrieved: List[RetrievedPassage] = field(default_factory=list)
    retrieval_enabled: bool = True
    plan: Optional[Plan] = None
    preferences: Dict[str, Pref] = field(default_factory=dict)   # final: explicit + inferred + unknown
    recommendations: List[Recommendation] = field(default_factory=list)
    explanation: Optional[Explanation] = None
    generator: str = ""                # who wrote the explanation prose
    generation_status: str = ""        # validation / fallback status line
    message: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)   # observable step log
