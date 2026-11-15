"""Unit tests for the LAMBADA accuracy scorer (iter 9).

The scoring surface is small enough that these tests cover the whole public API
without touching the engine or the dataset file: the normalisation rules, the
first-word extraction, and the end-to-end score() verdict for the handful of
cases that distinguish a correctly-decoding model from a broken one. The
dataset reader is covered by a tmp_path fixture that writes a two-line
mini-JSONL, so load_lambada's splitting and limit handling are exercised
without pulling the real 2 MB file into the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmopt.accuracy import (
    LambadaSample,
    first_word,
    load_lambada,
    normalise,
    score,
)


class TestNormalise:
    def test_lowercases(self):
        assert normalise("Signs") == "signs"

    def test_strips_trailing_punctuation(self):
        assert normalise("signs.") == "signs"
        assert normalise("signs!") == "signs"
        assert normalise("signs?") == "signs"
        assert normalise("signs,") == "signs"
        assert normalise("signs:") == "signs"
        assert normalise("signs;") == "signs"
        assert normalise("signs)") == "signs"
        assert normalise('signs"') == "signs"

    def test_keeps_internal_apostrophe(self):
        # LAMBADA gold words include contractions
        assert normalise("don't") == "don't"
        assert normalise("Don't") == "don't"

    def test_keeps_internal_hyphen(self):
        assert normalise("half-blood") == "half-blood"

    def test_strips_surrounding_whitespace(self):
        assert normalise("  signs  ") == "signs"


class TestFirstWord:
    def test_simple(self):
        assert first_word("signs and wonders") == "signs"

    def test_leading_whitespace(self):
        # vLLM completions often start with a leading space before the first token
        assert first_word(" signs and wonders") == "signs"

    def test_single_word(self):
        assert first_word("signs") == "signs"

    def test_empty(self):
        assert first_word("") == ""
        assert first_word("   ") == ""


class TestScore:
    def test_exact_match(self):
        assert score(" signs and wonders", "signs") is True

    def test_case_insensitive(self):
        assert score(" Signs and wonders", "signs") is True

    def test_trailing_punct_in_response(self):
        assert score(" signs. And wonders", "signs") is True

    def test_mismatch(self):
        assert score(" wonders and signs", "signs") is False

    def test_empty_response_fails(self):
        assert score("", "signs") is False
        assert score("   ", "signs") is False

    def test_catches_iter7_repetition_garbage(self):
        """The exact failure mode iter 7 produced on Qwen3-8B greedy decode."""
        garbage = " of the of of of of of of the of of of"
        assert score(garbage, "signs") is False


class TestLoadLambada:
    def test_round_trip(self, tmp_path: Path):
        src = tmp_path / "mini.jsonl"
        src.write_text(
            json.dumps({"text": "The man walked across the signs"}) + "\n"
            + json.dumps({"text": "She said hello"}) + "\n",
            encoding="utf-8",
        )
        samples = list(load_lambada(src))
        assert len(samples) == 2
        assert samples[0] == LambadaSample(
            prompt="The man walked across the", gold="signs"
        )
        assert samples[1] == LambadaSample(prompt="She said", gold="hello")

    def test_limit(self, tmp_path: Path):
        src = tmp_path / "mini.jsonl"
        src.write_text(
            "\n".join(json.dumps({"text": f"row {i} word{i}"}) for i in range(10))
            + "\n",
            encoding="utf-8",
        )
        samples = list(load_lambada(src, limit=3))
        assert len(samples) == 3
        assert samples[0].gold == "word0"
        assert samples[2].gold == "word2"

    def test_skips_blank_lines(self, tmp_path: Path):
        src = tmp_path / "mini.jsonl"
        src.write_text(
            json.dumps({"text": "a b"}) + "\n\n"
            + json.dumps({"text": "c d"}) + "\n",
            encoding="utf-8",
        )
        samples = list(load_lambada(src))
        assert len(samples) == 2

    def test_skips_single_word_rows(self, tmp_path: Path):
        src = tmp_path / "mini.jsonl"
        src.write_text(
            json.dumps({"text": "singleton"}) + "\n"
            + json.dumps({"text": "a b"}) + "\n",
            encoding="utf-8",
        )
        samples = list(load_lambada(src))
        assert len(samples) == 1
        assert samples[0].gold == "b"


class TestRealDatasetSmoke:
    """One smoke test that pulls the real 5153-sample file (committed to samples/)
    just to confirm the format on disk hasn't drifted from what the scorer expects."""

    def test_real_file_loads(self):
        path = Path(__file__).resolve().parents[2] / "samples" / "lambada_test.jsonl"
        if not path.exists():
            pytest.skip(f"{path} not downloaded")
        samples = list(load_lambada(path, limit=5))
        assert len(samples) == 5
        for s in samples:
            assert s.prompt
            assert s.gold
            assert " " in s.prompt or len(s.prompt) > 0
