"""Inspector-gated evidence selection for the MAPPO rollout.

RECONSTRUCTED 2026-08-19. The original file was truncated to 0 bytes when /data
filled up during the v13 run (crash at step 19, file mtime 18:42 vs crash
18:36); it was untracked by git and no copy survived on disk. This version was
rebuilt to satisfy every case in ``unittest/test_inspector_filter.py`` plus the
two call sites (``trainer_vidorag_restore._resolve_synth_evidence_pages`` and
``si_rewards.compute_inspector_ref_reward``). Behaviour is pinned by that test
file, which covers each route and each malformed-input branch.

WHAT THIS DOES
    In the three-agent rollout the Seeker picks a handful of candidate pages and
    the (frozen) Inspector then judges them. Its ``reference`` field names the
    subset that actually supports the answer. With
    ``VIDORAG_INSPECTOR_FILTER=1`` the Synthesizer is fed only that subset
    instead of the Seeker's full selection, so the Inspector acts as a learned
    evidence gate.

    ``reference`` values are 0-BASED POSITIONS IN THE SEEKER'S SELECTION LIST,
    not candidate-pool indices and not page numbers. With a selection of
    ``[2, 5, 7]``, ``"reference": [0, 2]`` means candidates 2 and 7.

THE ONE INVARIANT THAT MATTERS
    A parse failure must NEVER shrink the evidence. Every unusable response
    falls back to the Seeker's full selection, which is exactly the pre-feature
    behaviour, so a flaky Inspector can only lose the *benefit* of filtering and
    can never starve the Synthesizer. In particular prose is never mined for
    digits: "The answer is on page 2." yields a fallback, not ``[2]``.

ROUTES
    ROUTE_SYNTHESIZER  usable reference -> filter to that subset
    ROUTE_SEEKER       the Inspector asked for more pages ("sufficient": false,
                       or the legacy information/choice shape). There is no
                       retrieval loop in the training rollout, so this is a dead
                       end for the filter and falls back; si_rewards penalises it
                       when it cost a gold page.
    ROUTE_NO_VALID_REF a reference field was present but nothing in it was a
                       valid position (empty list, all out of range, all junk)
    ROUTE_UNPARSED     no JSON object, or JSON without a reference field and
                       without a seeker signal
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROUTE_SYNTHESIZER = "synthesizer"
ROUTE_SEEKER = "seeker"
# Value pinned to "no_valid_reference" (NOT "no_valid_ref") by the logs of the
# runs that used the original module: v13_run100 steps 0-19, inspfilter_run50 and
# inspfilter_run100 all print `route=no_valid_reference`. The test file only
# compares against the imported constant, so it does not constrain this string;
# keeping it identical is what lets one log-parser read the whole run.
ROUTE_NO_VALID_REF = "no_valid_reference"
ROUTE_UNPARSED = "unparsed"

ENV_INSPECTOR_FILTER = "VIDORAG_INSPECTOR_FILTER"

# Only these spellings count as "the Inspector said no". Anything else in a
# `sufficient` field is treated as truthy so that an unexpected value degrades
# towards the answer path rather than towards a spurious seeker route.
_FALSEY_STRINGS = frozenset({"false", "no", "0", "none", "null", ""})

# `,}` / `,]` — the one JSON deviation small instruct models emit often enough
# to be worth repairing rather than dropping the whole episode.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def inspector_filter_enabled() -> bool:
    """True only when ``VIDORAG_INSPECTOR_FILTER`` is exactly ``"1"``.

    Deliberately strict: "true"/"yes" do NOT enable it. This flag decides
    whether a run is comparable to the unfiltered baselines, so a typo must
    fail closed (historical path) rather than silently switch the experiment.
    """
    return os.environ.get(ENV_INSPECTOR_FILTER, "").strip() == "1"


def _iter_json_objects(text: str):
    """Yield candidate JSON-object substrings, outermost-first.

    Brace-balanced and string-aware, so prose before/after the object and
    braces inside string values are both handled. A truncated object yields
    nothing (no closing brace is ever found).
    """
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def _load_inspector_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort decode of the Inspector's response into a dict.

    Returns None when nothing dict-shaped can be recovered. Never guesses at
    content: the only repair applied is trailing-comma removal.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    body = text
    fenced = _FENCE_RE.search(text)
    if fenced is not None:
        body = fenced.group(1)

    for chunk in _iter_json_objects(body):
        for candidate in (chunk, _TRAILING_COMMA_RE.sub(r"\1", chunk)):
            try:
                obj = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _is_falsey(value: Any) -> bool:
    """Whether a ``sufficient`` value means "not sufficient"."""
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in _FALSEY_STRINGS
    if isinstance(value, int):
        return value == 0
    return False


def _coerce_position(value: Any) -> Optional[int]:
    """An index, or None if this value is not one.

    Accepts ints and integer-valued strings (the model quotes them often
    enough). Rejects bools (``True`` is an ``int`` in Python and would silently
    become position 1), floats, and any string that is not purely an integer —
    "page 2" must not become 2.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _valid_positions(raw: Any, n_selection: int) -> List[int]:
    """Filter ``raw`` down to in-range positions, keeping the model's order.

    De-duplicates (``"1"`` and ``1`` collapse) and drops anything outside
    ``[0, n_selection)``.
    """
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raw = [raw]  # scalar reference, e.g. "reference": 1

    positions: List[int] = []
    seen = set()
    for item in raw:
        pos = _coerce_position(item)
        if pos is None or pos < 0 or pos >= n_selection or pos in seen:
            continue
        seen.add(pos)
        positions.append(pos)
    return positions


def parse_inspector_reference(
    text: Optional[str], n_selection: int
) -> Tuple[List[int], str]:
    """Parse the Inspector response into ``(positions, route)``.

    Args:
        text: the Inspector's decoded response.
        n_selection: how many pages the Seeker handed over. Positions outside
            ``[0, n_selection)`` are dropped.

    Returns:
        ``(positions, route)``. ``positions`` is non-empty ONLY for
        ``ROUTE_SYNTHESIZER``; every other route returns ``[]`` and means
        "use the full selection".
    """
    obj = _load_inspector_json(text)
    if obj is None:
        return [], ROUTE_UNPARSED

    # An explicit "not sufficient" wins over anything else, including a
    # reference the model emitted anyway: it is asking for more pages.
    if "sufficient" in obj and _is_falsey(obj["sufficient"]):
        return [], ROUTE_SEEKER

    # Legacy feedback shape: {"information": ..., "choice": [...], "reason": ...}
    if "choice" in obj or "information" in obj:
        return [], ROUTE_SEEKER

    if "reference" not in obj:
        missing = obj.get("missing")
        if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes)):
            if len(missing) > 0:
                return [], ROUTE_SEEKER
        elif missing:
            return [], ROUTE_SEEKER
        return [], ROUTE_UNPARSED

    positions = _valid_positions(obj["reference"], n_selection)
    if not positions:
        return [], ROUTE_NO_VALID_REF
    return positions, ROUTE_SYNTHESIZER


def select_evidence_indices(
    selection: Sequence[int], text: Optional[str]
) -> Tuple[List[int], Dict[str, Any]]:
    """Map the Inspector's reference onto the Seeker's selection.

    Args:
        selection: candidate-pool indices the Seeker chose, in Seeker order.
        text: the Inspector's decoded response.

    Returns:
        ``(evidence, info)``. ``evidence`` is always a NEW list — never the
        caller's object — of candidate-pool indices for the Synthesizer.
        ``info`` carries ``route``, ``n_seeker``, ``n_inspector``, ``fallback``
        and ``filter_ratio`` (nan when the Seeker selected nothing).

    On any unusable response ``evidence`` is a copy of the full selection and
    ``fallback`` is True. Note that an Inspector which legitimately keeps every
    page is NOT a fallback: ``filter_ratio == 1.0`` with ``fallback False``, so
    the two cases stay distinguishable in the logs.
    """
    sel: List[int] = list(selection)
    n_seeker = len(sel)
    positions, route = parse_inspector_reference(text, n_seeker)

    if positions:
        evidence = [sel[p] for p in positions]
        fallback = False
    else:
        evidence = list(sel)
        fallback = True

    info: Dict[str, Any] = {
        "route": route,
        "n_seeker": n_seeker,
        "n_inspector": len(evidence),
        "fallback": bool(fallback),
        "filter_ratio": (len(evidence) / n_seeker) if n_seeker else math.nan,
    }
    return evidence, info
