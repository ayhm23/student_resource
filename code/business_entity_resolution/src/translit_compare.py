"""Step 3 Track A2: compare unidecode vs. indic_transliteration on train GT pairs.

For every train matched pair where exactly one side's name is non-Latin,
transliterates the non-Latin side with both methods, builds a consonant
skeleton from each result, and measures the % of pairs whose skeleton token
sets overlap with the Latin side's skeleton token set by >= 50%. Reports a
per-script winner, used to decide which method `normalize.py` should use for
each script going forward.
"""

import re
import sys
from collections import Counter, defaultdict

import duckdb
from unidecode import unidecode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from config import RAW_DIR
from normalize import WORD_RE
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F, sanscript.DEVANAGARI),
    ("Bengali", 0x0980, 0x09FF, sanscript.BENGALI),
    ("Gurmukhi", 0x0A00, 0x0A7F, sanscript.GURMUKHI),
    ("Gujarati", 0x0A80, 0x0AFF, sanscript.GUJARATI),
    ("Odia", 0x0B00, 0x0B7F, sanscript.ORIYA),
    ("Tamil", 0x0B80, 0x0BFF, sanscript.TAMIL),
    ("Telugu", 0x0C00, 0x0C7F, sanscript.TELUGU),
    ("Kannada", 0x0C80, 0x0CFF, sanscript.KANNADA),
    ("Malayalam", 0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
SCRIPT_BY_NAME = {name: scheme for name, _, _, scheme in SCRIPT_RANGES}

CONSONANT_SUBS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"),
                   ("gh", "g"), ("ch", "c"), ("sh", "s"), ("z", "j"), ("q", "k"),
                   ("w", "v"), ("ck", "k")]
VOWEL_HY_RE = re.compile(r"[aeiouhy]")
REPEAT_RE = re.compile(r"(.)\1+")


def detect_script(text):
    """Return the dominant Indic script name in ``text``, else None (Latin/other)."""
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for name, lo, hi, _ in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def skeleton_token(tok):
    """Build the consonant skeleton of one lowercase-ASCII token per the Track A2 spec."""
    if not tok:
        return ""
    s = tok.lower()
    for a, b in CONSONANT_SUBS:
        s = s.replace(a, b)
    first, rest = s[0], s[1:]
    rest = VOWEL_HY_RE.sub("", rest)
    skel = first + rest
    return REPEAT_RE.sub(r"\1", skel)


def skeleton_set(text):
    """Tokenize ``text`` and return the set of skeletonized tokens."""
    return {skeleton_token(t) for t in WORD_RE.findall(text.lower())}


def overlap_ge_half(set_a, set_b):
    """True if the smaller set's overlap fraction with the other is >= 50%."""
    if not set_a or not set_b:
        return False
    inter = len(set_a & set_b)
    return inter / min(len(set_a), len(set_b)) >= 0.5


def main():
    """Compare unidecode vs indic_transliteration per script on train GT pairs."""
    print_sysinfo()
    con = duckdb.connect()
    path = (RAW_DIR / "train_gt_joined.parquet").as_posix()

    with stage("scan train_gt_joined for non-Latin/Latin name pairs"):
        rows = con.execute(f"SELECT s1_name, match_name FROM '{path}'").fetchall()
    print(f"  scanned {len(rows)} pairs")

    per_script = defaultdict(lambda: {"n": 0, "unidecode_hits": 0, "indic_hits": 0})
    for s1_name, match_name in rows:
        s1_script = detect_script(s1_name)
        m_script = detect_script(match_name)
        if bool(s1_script) == bool(m_script):
            continue  # need exactly one non-Latin side
        non_latin_text, latin_text, script = (s1_name, match_name, s1_script) if s1_script else (match_name, s1_name, m_script)
        if not non_latin_text or not latin_text:
            continue
        latin_skel = skeleton_set(latin_text)

        uni_text = unidecode(non_latin_text)
        uni_skel = skeleton_set(uni_text)

        try:
            indic_text = transliterate(non_latin_text, SCRIPT_BY_NAME[script], sanscript.ITRANS)
        except Exception:
            indic_text = uni_text
        indic_skel = skeleton_set(indic_text)

        d = per_script[script]
        d["n"] += 1
        if overlap_ge_half(uni_skel, latin_skel):
            d["unidecode_hits"] += 1
        if overlap_ge_half(indic_skel, latin_skel):
            d["indic_hits"] += 1

    print("\n=== Skeleton-overlap>=50% rate by script and method ===")
    print(f"{'script':<12}{'n':>8}{'unidecode':>12}{'indic_translit':>16}{'winner':>18}")
    winners = {}
    for script, d in sorted(per_script.items(), key=lambda kv: -kv[1]["n"]):
        n = d["n"]
        uni_rate = d["unidecode_hits"] / n if n else 0
        indic_rate = d["indic_hits"] / n if n else 0
        winner = "indic_transliteration" if indic_rate >= uni_rate else "unidecode"
        winners[script] = winner
        print(f"{script:<12}{n:>8}{uni_rate*100:>11.1f}%{indic_rate*100:>15.1f}%{winner:>18}")

    print("\nWINNERS_DICT =", {k: v for k, v in winners.items()})
    con.close()


if __name__ == "__main__":
    main()
