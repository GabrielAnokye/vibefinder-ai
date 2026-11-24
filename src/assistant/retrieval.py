"""
Local multi-source retrieval: Markdown knowledge guides + catalog-derived facts.

- Guides are split on '## ' headings. A heading may end with '{#id}' to fix its id; otherwise
  the heading is slugified. Source ids are '<file stem>#<section id>', e.g. listening_contexts#studying.
- A section may contain one 'Hints: `k=v; ...`' line and one 'Applies when: `k=v`' line.
  Only whitelisted keys with valid values are kept; everything else is recorded as rejected.
  Section prose is data for citation, never instructions.
- Relevance is TF-IDF cosine similarity (standard library only). Ties break by source id,
  so results are deterministic.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.assistant.catalog import catalog_passages, slugify
from src.assistant.models import CONTEXTS, Passage, RetrievedPassage
from src.recommender import MOOD_TAGS, STRATEGIES, normalize_label

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

GUIDE_RESULTS = 3
CATALOG_RESULTS = 3
MIN_SCORE = 0.08

HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*(?:\{#(?P<id>[a-z0-9-]+)\})?\s*$")
HINT_LINE = re.compile(r"^Hints:\s*`(?P<body>[^`]*)`\s*$")
WHEN_LINE = re.compile(r"^Applies when:\s*`(?P<body>[^`]*)`\s*$")

STOPWORDS = frozenset("""
a an and any are as at be but by can do does exactly for from get give have i if in into is it its
just me my need of on only or our please recommend recommendation recommendations should so some
something song songs than that the their them then there these this to too track tracks up
us want was we what when which while who will with would you your
""".split())


# ---------------------------------------------------------------- hint validation (whitelist)

def _unit(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("must be between 0 and 1")
    return number


def _boolean(value: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError("must be true or false")
    return value == "true"


def _popularity(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 100:
        raise ValueError("must be between 0 and 100")
    return number


def _tags(value: str) -> Tuple[str, ...]:
    tags = tuple(dict.fromkeys(normalize_label(t) for t in value.split(",") if t.strip()))
    if not tags or any(t not in MOOD_TAGS for t in tags):
        raise ValueError(f"tags must come from {sorted(MOOD_TAGS)}")
    return tags


def _mode(value: str) -> str:
    if value not in STRATEGIES:
        raise ValueError(f"mode must be one of {list(STRATEGIES)}")
    return value


def _hint_parsers(valid_moods: Iterable[str]) -> Dict[str, Any]:
    moods = frozenset(valid_moods)

    def _moods(value: str) -> Tuple[str, ...]:
        found = tuple(dict.fromkeys(normalize_label(m) for m in value.split(",") if m.strip()))
        if not found or any(m not in moods for m in found):
            raise ValueError(f"moods must come from the catalog: {sorted(moods)}")
        return found

    return {
        "energy": _unit, "acoustic": _boolean, "instrumentalness": _unit,
        "vocal_intensity": _unit, "popularity": _popularity, "mood_tags": _tags,
        "moods": _moods, "diversity": _boolean, "mode": _mode,
    }


ALLOWED_HINT_KEYS = ("energy", "acoustic", "instrumentalness", "vocal_intensity", "popularity",
                     "mood_tags", "moods", "diversity", "mode")
ALLOWED_WHEN_KEYS = ("context", "unavailable_genre")


def parse_hints(body: str, valid_moods: Iterable[str]) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """Parse 'k=v; k=v' into whitelisted, validated pairs; return (accepted, rejected descriptions)."""
    parsers = _hint_parsers(valid_moods)
    accepted, rejected = [], []
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        key, sep, raw = part.partition("=")
        key, raw = key.strip().casefold(), raw.strip().casefold()
        if not sep or key not in parsers:
            rejected.append(f"{part!r}: key not allowed")
            continue
        try:
            accepted.append((key, parsers[key](raw)))
        except ValueError as error:
            rejected.append(f"{part!r}: {error}")
    return accepted, rejected


def parse_when(body: str) -> Tuple[List[Tuple[str, Tuple[str, ...]]], List[str]]:
    accepted, rejected = [], []
    for part in body.split(";"):
        key, sep, raw = part.strip().partition("=")
        key = key.strip().casefold()
        values = tuple(normalize_label(v) for v in raw.split("|") if v.strip())
        if not sep or key not in ALLOWED_WHEN_KEYS or not values:
            rejected.append(f"applies-when {part.strip()!r}: not allowed")
        elif key == "context" and any(v not in CONTEXTS for v in values):
            rejected.append(f"applies-when {part.strip()!r}: unknown context")
        else:
            accepted.append((key, values))
    return accepted, rejected


# ---------------------------------------------------------------- loading guides

def load_guide(path: Path, valid_moods: Iterable[str]) -> List[Passage]:
    """Split one Markdown guide into passages with stable ids."""
    stem = path.stem
    passages: List[Passage] = []
    current: Optional[Dict[str, Any]] = None

    def finish() -> None:
        if current is None:
            return
        text = " ".join(" ".join(current["lines"]).split())
        passages.append(Passage(
            source_id=f"{stem}#{current['id']}", title=current["title"], text=text, origin="guide",
            hints=tuple(current["hints"]), applies_when=tuple(current["when"]),
            rejected_hints=tuple(current["rejected"]),
        ))

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING.match(line)
        if heading:
            finish()
            section_id = heading.group("id") or slugify(heading.group("title"))
            current = {"id": section_id, "title": heading.group("title"), "lines": [],
                       "hints": [], "when": [], "rejected": []}
            continue
        if current is None:
            continue  # document preamble is not a retrievable section
        hint, when = HINT_LINE.match(line.strip()), WHEN_LINE.match(line.strip())
        if hint:
            accepted, rejected = parse_hints(hint.group("body"), valid_moods)
            current["hints"].extend(accepted)
            current["rejected"].extend(rejected)
        elif when:
            accepted, rejected = parse_when(when.group("body"))
            current["when"].extend(accepted)
            current["rejected"].extend(rejected)
        else:
            current["lines"].append(line)
    finish()
    return passages


def load_knowledge(songs: List[Dict], knowledge_dir: Path = KNOWLEDGE_DIR) -> List[Passage]:
    """All guide sections (sorted by file name) plus catalog-derived passages."""
    valid_moods = {normalize_label(s["mood"]) for s in songs}
    passages: List[Passage] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        passages.extend(load_guide(path, valid_moods))
    passages.extend(catalog_passages(songs))
    ids = [p.source_id for p in passages]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate knowledge source ids: {', '.join(duplicates)}")
    return passages


# ---------------------------------------------------------------- TF-IDF retrieval

def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def terms(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def _weights(counts: Counter, idf: Dict[str, float]) -> Dict[str, float]:
    vector = {t: (1.0 + math.log(n)) * idf[t] for t, n in counts.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in vector.values()))
    return {t: v / norm for t, v in vector.items()} if norm else {}


class Retriever:
    """TF-IDF index over passages. Build once, query many times."""

    def __init__(self, passages: Sequence[Passage]):
        self.passages = list(passages)
        self.by_id = {p.source_id: p for p in self.passages}
        docs = [Counter(terms(f"{p.title} {p.text}")) for p in self.passages]
        n = len(docs)
        df = Counter(t for doc in docs for t in doc)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.vectors = [_weights(doc, self.idf) for doc in docs]

    def search(self, query: str, guide_results: int = GUIDE_RESULTS,
               catalog_results: int = CATALOG_RESULTS, min_score: float = MIN_SCORE) -> List[RetrievedPassage]:
        """Top passages per source type above min_score, ordered by score then source id."""
        q = _weights(Counter(terms(query)), self.idf)
        if not q:
            return []
        scored = []
        for passage, vector in zip(self.passages, self.vectors):
            score = sum(weight * vector.get(t, 0.0) for t, weight in q.items())
            if score >= min_score:
                scored.append(RetrievedPassage(passage, round(score, 4)))
        scored.sort(key=lambda r: (-r.score, r.passage.source_id))
        limits = {"guide": guide_results, "catalog": catalog_results}
        taken: Dict[str, int] = Counter()
        results = []
        for r in scored:
            if taken[r.passage.origin] < limits.get(r.passage.origin, 0):
                taken[r.passage.origin] += 1
                results.append(r)
        return results
