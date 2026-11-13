"""
Turning an LLM conclusion into structured, checkable claims.

A claim is (feature, direction): "feature X increases / decreases the ionization energy".
The conclude prompt asks the LLM to end its answer with a line
    CLAIMS: [{"feature": "fluorine", "effect": "increases"}, ...]
using only names of the closed vocabulary (features.py). If the block is missing or
cannot be parsed (small local models), a conservative keyword parser is used: a sentence
gives a claim only if it names exactly one feature and exactly one direction.
"""
from __future__ import annotations

import json
import re

from .features import FEATURES, FEATURE_NAMES, vocabulary_for_prompt

UP_WORDS = ("increase", "raise", "higher", "enhance", "boost", "improv", "elevat", "stabiliz")
DOWN_WORDS = ("decrease", "lower", "reduc", "diminish", "drop", "destabiliz", "worsen")

_CLAIMS_RE = re.compile(r"CLAIMS\s*:\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)


def claims_instruction() -> str:
    """Text appended to the conclude prompt (identical in every memory mode)."""
    return (
        "\n\nAt the very end, add ONE line starting with 'CLAIMS:' followed by a JSON list "
        "of the structure-property relationships you believe in, for example "
        'CLAIMS: [{"feature": "fluorine", "effect": "increases"}]. '
        "Use 'increases' or 'decreases' (effect on the ionization energy) and ONLY these "
        f"feature names: {', '.join(FEATURE_NAMES)}. Use an empty list if you have none."
    )


def strip_claims_block(text: str) -> str:
    """Conclusion text without the CLAIMS line (what a human would call the lesson)."""
    return _CLAIMS_RE.sub("", text or "").strip()


def _norm_direction(value: str) -> str | None:
    v = (value or "").lower()
    if any(w in v for w in UP_WORDS) or v in ("up", "+", "positive"):
        return "up"
    if any(w in v for w in DOWN_WORDS) or v in ("down", "-", "negative"):
        return "down"
    return None


def _norm_feature(value: str) -> str | None:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in FEATURES:
        return v
    for name in FEATURE_NAMES:                       # tolerate "fluorine atoms", "CF3"...
        if any(k.replace(" ", "_") in v for k in FEATURES[name][2]):
            return name
    return None


def parse_claims_block(text: str) -> list[dict] | None:
    """Claims from the CLAIMS JSON block, or None if there is no parsable block."""
    m = _CLAIMS_RE.search(text or "")
    if not m:
        return None
    try:
        items = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        feat = _norm_feature(str(it.get("feature", "")))
        direction = _norm_direction(str(it.get("effect", it.get("direction", ""))))
        if feat and direction:
            out.append({"feature": feat, "direction": direction, "source": "json"})
    return out


def _features_in(sentence: str) -> list[str]:
    s = sentence.lower()
    found = []
    for name in FEATURE_NAMES:                       # specific names first ("trifluoromethyl" before "methyl")
        for kw in FEATURES[name][2]:
            if kw in s:
                found.append(name)
                s = s.replace(kw, " ")
                break
    return found


def parse_claims_keywords(text: str) -> list[dict]:
    out = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", strip_claims_block(text)):
        feats = _features_in(sentence)
        s = sentence.lower()
        ups = any(w in s for w in UP_WORDS)
        downs = any(w in s for w in DOWN_WORDS)
        if len(feats) == 1 and ups != downs:
            out.append({"feature": feats[0], "direction": "up" if ups else "down",
                        "source": "keywords", "sentence": sentence.strip()[:300]})
    return out


def extract_claims(text: str) -> list[dict]:
    """Structured claims of a conclusion, deduplicated by (feature, direction)."""
    claims = parse_claims_block(text)
    if claims is None:
        claims = parse_claims_keywords(text)
    seen, out = set(), []
    for c in claims:
        key = (c["feature"], c["direction"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def claim_id(feature: str, direction: str) -> str:
    return f"{feature}:{direction}"


def claim_sentence(feature: str, direction: str) -> str:
    verb = "increases" if direction == "up" else "decreases"
    return f"{FEATURES[feature][1] if feature in FEATURES else feature} {verb} the ionization energy"


__all__ = ["claims_instruction", "strip_claims_block", "extract_claims", "claim_id",
           "claim_sentence", "vocabulary_for_prompt"]
