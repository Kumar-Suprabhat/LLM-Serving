"""Accuracy scoring for the LAMBADA last-word-prediction benchmark (iter 9).

The project's performance iterations (iter 7 FP8 KV, iter 8 weight quantisation,
later retries) all introduce *lossy* arithmetic to buy throughput or memory, and
all have a quality cost the measurement path must detect. LAMBADA is the smallest
useful benchmark that catches it: each sample is a passage whose last word the
model has to predict from everything before. Greedy decode makes the test
deterministic; exact-match on the first predicted word is a continuous signal
that collapses to zero the moment a quantisation step breaks the model (iter 7's
repetition garbage would score close to 0% here).

The module is intentionally kept small — scoring logic and a dataset reader only.
The engine-side driver lives in ``scripts/remote/accuracy.py`` and reuses the
same :class:`CompletionsClient` the perf benches already use, so an accuracy run
is just "another arm" alongside each perf arm in ``runs/`` with the same
Parquet + JSON layout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# A trailing span the LAMBADA scorer is willing to strip from the predicted
# first word before comparing. Keeps internal apostrophes (``didn't``) intact
# because stripping them would make the normalisation lossy — gold words in
# LAMBADA include contractions and the model emits them verbatim when correct.
_TRAILING_PUNCT = re.compile(r"[\.,!?;:\)\]\"'`]+$")


@dataclass(frozen=True, slots=True)
class LambadaSample:
    """One LAMBADA passage, split into the prompt and the single gold last word.

    ``prompt`` is everything the model gets to see; ``gold`` is the last word
    that was removed from the original passage. Both are already stripped of
    the whitespace that separated them in the source, so the prompt never ends
    in a trailing space (vLLM would otherwise emit a leading space in the
    completion and the scorer would see the right word a byte later).
    """

    prompt: str
    gold: str


def load_lambada(path: Path, *, limit: int | None = None) -> Iterator[LambadaSample]:
    r"""Yield :class:`LambadaSample`\s from a LAMBADA JSONL file in order.

    The source format is one JSON object per line with a single ``text`` field;
    the gold last word is the span after the last ASCII space in that text. A
    row whose text has no space (one-word passage) or whose last word is empty
    after stripping is dropped — it would not be a valid LAMBADA sample.
    """
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        if " " not in text:
            continue
        prompt, gold = text.rsplit(" ", 1)
        gold = gold.strip()
        if not gold:
            continue
        yield LambadaSample(prompt=prompt, gold=gold)
        if limit is not None and i + 1 >= limit:
            return


def normalise(word: str) -> str:
    """Return the comparable form of a word: lowercased, trailing punctuation stripped.

    Case is folded because LAMBADA gold words are lowercase but a decoder may
    produce a capitalised form at sentence start. Internal apostrophes and
    hyphens are kept — ``don't`` and ``half-blood`` are legitimate gold words
    whose middle characters carry meaning.
    """
    return _TRAILING_PUNCT.sub("", word.strip()).lower()


def first_word(text: str) -> str:
    """Return the first whitespace-delimited token of ``text``, or ``""`` if none.

    Used to extract the model's prediction from a greedy completion: the model
    may continue past the first word, but LAMBADA only scores the single word
    the prompt was truncated at.
    """
    for token in text.split():
        return token
    return ""


def score(response_text: str, gold: str) -> bool:
    """Return True iff the model's first word matches the gold word under :func:`normalise`.

    Empty responses never match — a model that returns no tokens has failed
    the task even if the gold word happens to be empty (which cannot occur
    because :func:`load_lambada` drops such rows).
    """
    predicted = first_word(response_text)
    if not predicted:
        return False
    return normalise(predicted) == normalise(gold)
