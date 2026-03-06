from __future__ import annotations

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

from .types import ProcessedDocument

DetectorFactory.seed = 0

EXPECTED_SCRIPTS = {
    "en": {"latin"},
    "es": {"latin"},
    "fr": {"latin"},
    "de": {"latin"},
    "ar": {"arabic"},
    "ru": {"cyrillic"},
    "zh": {"cjk"},
    "ja": {"cjk", "hiragana", "katakana"},
    "ko": {"hangul", "cjk"},
    "hi": {"devanagari"},
}


def _script_of_char(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    if "HANGUL" in name:
        return "hangul"
    if "CJK" in name or "IDEOGRAPH" in name:
        return "cjk"
    if "DEVANAGARI" in name:
        return "devanagari"
    return ""


def _script_match_ratio(text: str, lang: str) -> tuple[float | None, int]:
    expected = EXPECTED_SCRIPTS.get(lang)
    if not expected:
        return None, 0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0, 0
    matches = sum(1 for ch in letters if _script_of_char(ch) in expected)
    return matches / len(letters), len(letters)


def _min_script_ratio(lang: str) -> float:
    by_lang = {
        "zh": 0.15,
        "ja": 0.2,
        "ko": 0.2,
        "hi": 0.25,
        "ar": 0.25,
        "ru": 0.25,
    }
    return by_lang.get(lang, 0.3)


def _canonical_lang(code: str | None) -> str:
    if not code:
        return ""
    value = str(code).strip().lower().replace("_", "-")
    if value.startswith("zh"):
        return "zh"
    for prefix in ("en", "es", "fr", "de", "ar", "ru", "ja", "ko", "hi"):
        if value.startswith(prefix):
            return prefix
    if "-" in value:
        return value.split("-", 1)[0]
    return value


def _select_detect_indices(
    docs: list[ProcessedDocument],
    *,
    max_detect_docs: int,
    seed: int,
) -> set[int]:
    if max_detect_docs <= 0 or max_detect_docs >= len(docs):
        return set(range(len(docs)))

    rng = random.Random(seed)
    per_lang_indices: dict[str, list[int]] = {}
    for idx, doc in enumerate(docs):
        per_lang_indices.setdefault(doc.lang, []).append(idx)

    for indices in per_lang_indices.values():
        rng.shuffle(indices)

    selected: set[int] = set()
    langs = sorted(per_lang_indices)
    if not langs:
        return selected

    base = max_detect_docs // len(langs)
    remainder = max_detect_docs % len(langs)

    leftovers: list[int] = []
    for lang in langs:
        indices = per_lang_indices[lang]
        target = base + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        take = min(len(indices), target)
        selected.update(indices[:take])
        leftovers.extend(indices[take:])

    if len(selected) < max_detect_docs and leftovers:
        rng.shuffle(leftovers)
        needed = max_detect_docs - len(selected)
        selected.update(leftovers[:needed])

    return selected


@dataclass
class LanguageAuditConfig:
    enabled: bool = True
    min_detect_chars: int = 80
    max_text_chars: int = 3000
    min_confidence: float = 0.85
    min_script_chars: int = 8
    max_detect_docs: int = 50000
    max_suspects: int = 5000
    drop_detected_mismatches: bool = False
    seed: int = 42


def run_language_audit(
    docs: list[ProcessedDocument],
    *,
    config: LanguageAuditConfig,
) -> tuple[list[ProcessedDocument], dict, list[dict]]:
    if not config.enabled:
        return docs, {
            "skipped": True,
            "reason": "Language audit disabled.",
            "input_docs": len(docs),
            "after_audit_docs": len(docs),
            "dropped_docs": 0,
        }, []

    detect_indices = _select_detect_indices(
        docs,
        max_detect_docs=max(0, int(config.max_detect_docs)),
        seed=config.seed,
    )

    kept_docs: list[ProcessedDocument] = []
    suspects: list[dict] = []
    retagged_pairs = Counter()

    overall = Counter()
    by_lang: dict[str, Counter] = {}

    for idx, doc in enumerate(docs):
        original_lang = doc.lang
        lang = original_lang
        ctr = by_lang.setdefault(lang, Counter())
        overall["total_docs"] += 1
        ctr["total_docs"] += 1

        text = doc.text.strip()
        script_ratio, script_letters = _script_match_ratio(text, lang)
        low_script = False
        if script_ratio is not None and script_letters >= config.min_script_chars:
            if script_ratio < _min_script_ratio(lang):
                low_script = True
                overall["low_script_docs"] += 1
                ctr["low_script_docs"] += 1

        detected_lang = ""
        detected_conf = None
        mismatch = False
        detect_error = False

        should_detect = idx in detect_indices
        if should_detect:
            overall["langdetect_inspected"] += 1
            ctr["langdetect_inspected"] += 1

            if len(text) < config.min_detect_chars:
                overall["langdetect_skipped_short"] += 1
                ctr["langdetect_skipped_short"] += 1
            else:
                snippet = text[: config.max_text_chars]
                try:
                    detections = detect_langs(snippet)
                    if detections:
                        detected_lang = _canonical_lang(detections[0].lang)
                        detected_conf = float(detections[0].prob)
                        if detected_conf >= config.min_confidence and detected_lang and detected_lang != lang:
                            mismatch = True
                            overall["langdetect_mismatch_docs"] += 1
                            ctr["langdetect_mismatch_docs"] += 1
                        else:
                            overall["langdetect_match_or_low_confidence_docs"] += 1
                            ctr["langdetect_match_or_low_confidence_docs"] += 1
                except LangDetectException:
                    detect_error = True
                    overall["langdetect_errors"] += 1
                    ctr["langdetect_errors"] += 1

        reasons: list[str] = []
        if mismatch:
            reasons.append("langdetect_mismatch")
        if low_script:
            reasons.append("low_expected_script_ratio")
        if detect_error:
            reasons.append("langdetect_error")

        if reasons:
            overall["suspicious_docs"] += 1
            ctr["suspicious_docs"] += 1
            if len(suspects) < config.max_suspects:
                suspects.append(
                    {
                        "raw_id": doc.raw_id,
                        "author_id": doc.author_id,
                        "source": doc.source,
                        "lang": original_lang,
                        "genre": doc.genre,
                        "token_length": doc.token_length,
                        "detected_lang": detected_lang or None,
                        "detected_confidence": detected_conf,
                        "script_ratio": script_ratio,
                        "reasons": reasons,
                        "text_preview": text[:300].replace("\n", " "),
                    }
                )

        drop = config.drop_detected_mismatches and mismatch
        if drop:
            overall["dropped_docs"] += 1
            ctr["dropped_docs"] += 1
            continue

        if mismatch and detected_lang and detected_lang != original_lang:
            doc.lang = detected_lang
            overall["retagged_docs"] += 1
            ctr["retagged_docs"] += 1
            retagged_pairs[f"{original_lang}->{detected_lang}"] += 1

        kept_docs.append(doc)

    inspected = overall.get("langdetect_inspected", 0)
    mismatches = overall.get("langdetect_mismatch_docs", 0)
    summary = {
        "skipped": False,
        "input_docs": len(docs),
        "after_audit_docs": len(kept_docs),
        "dropped_docs": overall.get("dropped_docs", 0),
        "detect_sample_docs": len(detect_indices),
        "langdetect_inspected": inspected,
        "langdetect_mismatch_docs": mismatches,
        "langdetect_mismatch_rate": (mismatches / inspected) if inspected else 0.0,
        "low_script_docs": overall.get("low_script_docs", 0),
        "suspicious_docs": overall.get("suspicious_docs", 0),
        "suspicious_rate": (overall.get("suspicious_docs", 0) / len(docs)) if docs else 0.0,
        "retagged_docs": overall.get("retagged_docs", 0),
        "retagged_pairs": dict(sorted(retagged_pairs.items())),
        "langdetect_errors": overall.get("langdetect_errors", 0),
        "langdetect_skipped_short": overall.get("langdetect_skipped_short", 0),
        "drop_detected_mismatches": config.drop_detected_mismatches,
        "thresholds": {
            "min_detect_chars": config.min_detect_chars,
            "max_text_chars": config.max_text_chars,
            "min_confidence": config.min_confidence,
            "min_script_chars": config.min_script_chars,
            "max_detect_docs": config.max_detect_docs,
            "max_suspects": config.max_suspects,
        },
        "by_lang": {
            lang: dict(sorted(counter.items())) for lang, counter in sorted(by_lang.items())
        },
    }
    return kept_docs, summary, suspects
