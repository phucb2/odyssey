"""Tests for the small English math lexical corpus generator."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from odyssey.experiments.math_corpus import (
    OPS,
    Rational,
    _apply_op,
    equation_domain,
    generate_dataset,
    iter_valid_equations,
    load_corpus,
    load_equations,
    rational_to_text,
    reduce_pair,
    render_sentences,
)


def test_reduce_pair_normalizes() -> None:
    assert reduce_pair(4, 8) == (1, 2)
    assert reduce_pair(0, 5) == (0, 1)


@pytest.mark.parametrize(
    ("num", "den", "expected"),
    [
        (1, 2, "one half"),
        (3, 4, "three fourths"),
        (2, 3, "two thirds"),
        (2, 5, "two fifths"),
        (2, 11, "two over eleven"),
    ],
)
def test_rational_to_text(num: int, den: int, expected: str) -> None:
    assert rational_to_text(num, den) == expected


def test_rational_to_text_article() -> None:
    assert rational_to_text(1, 2, article=True) == "a half"
    assert rational_to_text(1, 3, article=True) == "a third"


def test_integer_rational_to_text() -> None:
    assert rational_to_text(5, 1) == "five"


def test_apply_op_minus_rejects_negative_result() -> None:
    lhs = Rational(2, 1)
    rhs = Rational(5, 1)
    assert _apply_op("minus", lhs, rhs) is None


def test_apply_op_divide_requires_exact_quotient() -> None:
    lhs = Rational(5, 1)
    rhs = Rational(2, 1)
    assert _apply_op("divide", lhs, rhs) is None

    exact = _apply_op("divide", Rational(6, 1), Rational(2, 1))
    assert exact == Rational(3, 1)


def test_apply_op_rational_divide() -> None:
    result = _apply_op("divide", Rational(1, 2), Rational(1, 4))
    assert result == Rational(2, 1)


def test_equation_domain() -> None:
    assert equation_domain(Rational(2, 1), Rational(3, 1)) == "I"
    assert equation_domain(Rational(1, 2), Rational(2, 1)) == "Q"


def test_render_sentences_contains_operators() -> None:
    sentences = render_sentences("plus", Rational(1, 1), Rational(1, 1), Rational(2, 1))
    assert any("plus" in s for s in sentences)
    assert any("equals" in s or "equal" in s or " is " in s for s in sentences)


def test_iter_valid_equations_respects_ops_and_domains() -> None:
    records = iter_valid_equations(max_int=5, max_den=4)
    assert records
    assert {record.op for record in records}.issubset(set(OPS))
    assert {record.domain for record in records}.issubset({"I", "Q"})

    for record in records:
        recomputed = _apply_op(record.op, record.lhs, record.rhs)
        assert recomputed == record.result
        if record.op == "minus":
            assert record.lhs.to_fraction() >= record.rhs.to_fraction()
        if record.op == "divide":
            assert record.rhs.to_fraction() != 0
            assert record.lhs.to_fraction() % record.rhs.to_fraction() == 0


def test_generate_dataset_writes_files(tmp_path: Path) -> None:
    root = generate_dataset(
        n_equations=10,
        max_int=5,
        max_den=4,
        seed=0,
        output=tmp_path,
    )

    assert (root / "corpus.txt").is_file()
    assert (root / "corpus_train.txt").is_file()
    assert (root / "equations.jsonl").is_file()
    assert (root / "equations_train.jsonl").is_file()
    assert (root / "equations_holdout.jsonl").is_file()
    assert (root / "manifest.json").is_file()

    corpus = load_corpus(root)
    assert corpus.strip()
    assert all(line.strip() for line in corpus.splitlines())

    equations = load_equations(root)
    assert len(equations) == 10
    assert {row["op"] for row in equations}.issubset(set(OPS))
    assert {row["domain"] for row in equations}.issubset({"I", "Q"})
    assert all(row["sentences"] for row in equations)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_equations"] == 10
    assert manifest["n_sentences"] > 10
    assert manifest["n_train_equations"] + manifest["n_holdout_equations"] == 10


def test_generate_dataset_reproducible(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    generate_dataset(n_equations=8, max_int=4, max_den=4, seed=7, output=root_a)
    generate_dataset(n_equations=8, max_int=4, max_den=4, seed=7, output=root_b)
    assert load_corpus(root_a) == load_corpus(root_b)


def test_rational_fraction_roundtrip() -> None:
    value = Rational.from_fraction(Fraction(3, 4))
    assert value.to_pair() == [3, 4]
