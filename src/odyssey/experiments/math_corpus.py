"""Generate a small English arithmetic lexical corpus (integers I and rationals Q)."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from odyssey.experiments.clip import int_to_text
from odyssey.paths import dataset_dir

DEFAULT_DATASET_NAME = "small"
OPS = ("plus", "minus", "multiply", "divide")
Domain = Literal["I", "Q"]

_ORDINALS: dict[int, tuple[str, str]] = {
    2: ("half", "halves"),
    3: ("third", "thirds"),
    4: ("fourth", "fourths"),
    5: ("fifth", "fifths"),
    6: ("sixth", "sixths"),
    7: ("seventh", "sevenths"),
    8: ("eighth", "eighths"),
    9: ("ninth", "ninths"),
    10: ("tenth", "tenths"),
}


@dataclass(frozen=True)
class Rational:
    """Reduced non-negative rational stored as numerator/denominator."""

    num: int
    den: int = 1

    def __post_init__(self) -> None:
        if self.den <= 0:
            raise ValueError(f"denominator must be positive, got {self.den}")
        if self.num < 0:
            raise ValueError(f"numerator must be non-negative, got {self.num}")
        reduced_num, reduced_den = reduce_pair(self.num, self.den)
        if reduced_num != self.num:
            object.__setattr__(self, "num", reduced_num)
        if reduced_den != self.den:
            object.__setattr__(self, "den", reduced_den)

    @classmethod
    def from_fraction(cls, value: Fraction) -> Rational:
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}")
        return cls(value.numerator, value.denominator)

    @classmethod
    def from_pair(cls, pair: tuple[int, int]) -> Rational:
        return cls.from_fraction(Fraction(pair[0], pair[1]))

    def to_fraction(self) -> Fraction:
        return Fraction(self.num, self.den)

    def to_pair(self) -> list[int]:
        frac = self.to_fraction()
        return [frac.numerator, frac.denominator]

    @property
    def is_integer(self) -> bool:
        return self.den == 1 or self.num % self.den == 0

    @property
    def domain(self) -> Domain:
        return "I" if self.is_integer else "Q"


def default_dataset_dir() -> Path:
    return dataset_dir(DEFAULT_DATASET_NAME)


def reduce_pair(num: int, den: int) -> tuple[int, int]:
    frac = Fraction(num, den)
    return frac.numerator, frac.denominator


def rational_to_text(num: int, den: int, *, article: bool = False) -> str:
    """Convert a reduced fraction to English words."""
    num, den = reduce_pair(num, den)
    if den == 1:
        return int_to_text(num)

    if den in _ORDINALS:
        singular, plural = _ORDINALS[den]
        if num == 1:
            return f"a {singular}" if article else f"one {singular}"
        return f"{int_to_text(num)} {plural if num != 1 else singular}"

    return f"{int_to_text(num)} over {int_to_text(den)}"


def value_to_text(value: Rational, *, article: bool = False) -> str:
    return rational_to_text(value.num, value.den, article=article)


def integer_operands(max_int: int) -> list[Rational]:
    if max_int < 0:
        raise ValueError(f"max_int must be non-negative, got {max_int}")
    return [Rational(n, 1) for n in range(max_int + 1)]


def rational_operands(max_den: int) -> list[Rational]:
    """Proper reduced fractions with denominator in 2..max_den."""
    if max_den < 2:
        raise ValueError(f"max_den must be at least 2, got {max_den}")

    out: list[Rational] = []
    seen: set[tuple[int, int]] = set()
    for den in range(2, max_den + 1):
        for num in range(1, den):
            if math.gcd(num, den) != 1:
                continue
            pair = reduce_pair(num, den)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(Rational(*pair))
    return out


def _apply_op(op: str, lhs: Rational, rhs: Rational) -> Rational | None:
    left = lhs.to_fraction()
    right = rhs.to_fraction()

    if op == "plus":
        result = left + right
    elif op == "minus":
        if left < right:
            return None
        result = left - right
    elif op == "multiply":
        result = left * right
    elif op == "divide":
        if right == 0:
            return None
        if left % right != 0:
            return None
        result = left / right
    else:
        raise ValueError(f"unknown op {op!r}")

    if result < 0:
        return None
    return Rational.from_fraction(result)


def equation_domain(lhs: Rational, rhs: Rational) -> Domain:
    return "Q" if lhs.domain == "Q" or rhs.domain == "Q" else "I"


def _plus_templates(a: str, b: str, c: str) -> list[str]:
    return [
        f"{a} plus {b} equals {c}",
        f"{a} plus {b} equal {c}",
        f"{a} and {b} make {c}",
        f"{a} added to {b} is {c}",
        f"the sum of {a} and {b} is {c}",
    ]


def _minus_templates(a: str, b: str, c: str) -> list[str]:
    return [
        f"{a} minus {b} equals {c}",
        f"{a} minus {b} equal {c}",
        f"{a} take away {b} is {c}",
        f"{a} subtract {b} equals {c}",
        f"{b} subtracted from {a} is {c}",
    ]


def _multiply_templates(a: str, b: str, c: str) -> list[str]:
    return [
        f"{a} times {b} equals {c}",
        f"{a} times {b} equal {c}",
        f"{a} multiplied by {b} equals {c}",
        f"the product of {a} and {b} is {c}",
    ]


def _divide_templates(a: str, b: str, c: str) -> list[str]:
    return [
        f"{a} divided by {b} equals {c}",
        f"{a} divided by {b} equal {c}",
        f"{a} over {b} equals {c}",
    ]


def render_sentences(op: str, lhs: Rational, rhs: Rational, result: Rational) -> list[str]:
    a = value_to_text(lhs, article=True)
    b = value_to_text(rhs, article=True)
    c = value_to_text(result)

    if op == "plus":
        templates = _plus_templates(a, b, c)
    elif op == "minus":
        templates = _minus_templates(a, b, c)
    elif op == "multiply":
        templates = _multiply_templates(a, b, c)
    elif op == "divide":
        templates = _divide_templates(a, b, c)
    else:
        raise ValueError(f"unknown op {op!r}")

    # Preserve order but drop accidental duplicates from article collisions.
    seen: set[str] = set()
    out: list[str] = []
    for sentence in templates:
        if sentence not in seen:
            seen.add(sentence)
            out.append(sentence)
    return out


@dataclass(frozen=True)
class EquationRecord:
    op: str
    domain: Domain
    lhs: Rational
    rhs: Rational
    result: Rational

    def to_json(self) -> dict:
        return {
            "op": self.op,
            "domain": self.domain,
            "lhs": self.lhs.to_pair(),
            "rhs": self.rhs.to_pair(),
            "result": self.result.to_pair(),
            "sentences": render_sentences(self.op, self.lhs, self.rhs, self.result),
        }


def iter_valid_equations(
    *,
    max_int: int = 20,
    max_den: int = 8,
) -> list[EquationRecord]:
    ints = integer_operands(max_int)
    fracs = rational_operands(max_den)
    operands = ints + fracs

    seen: set[tuple[str, tuple[int, int], tuple[int, int]]] = set()
    records: list[EquationRecord] = []

    for lhs in operands:
        for rhs in operands:
            for op in OPS:
                result = _apply_op(op, lhs, rhs)
                if result is None:
                    continue
                key = (op, lhs.to_pair(), rhs.to_pair())
                key_tuple = (key[0], tuple(key[1]), tuple(key[2]))
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                records.append(
                    EquationRecord(
                        op=op,
                        domain=equation_domain(lhs, rhs),
                        lhs=lhs,
                        rhs=rhs,
                        result=result,
                    )
                )
    return records


def equation_key(record: EquationRecord | dict) -> tuple[str, tuple[int, int], tuple[int, int]]:
    if isinstance(record, dict):
        return (record["op"], tuple(record["lhs"]), tuple(record["rhs"]))
    return (record.op, tuple(record.lhs.to_pair()), tuple(record.rhs.to_pair()))


def split_equations(
    records: list[EquationRecord],
    *,
    holdout_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[EquationRecord], list[EquationRecord]]:
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(len(shuffled) * holdout_frac))) if shuffled else 0
    if n_holdout >= len(shuffled):
        n_holdout = max(1, len(shuffled) // 10)
    holdout = shuffled[:n_holdout]
    train = shuffled[n_holdout:]
    return train, holdout


def records_to_sentences(records: list[EquationRecord]) -> list[str]:
    sentences: list[str] = []
    for record in records:
        sentences.extend(render_sentences(record.op, record.lhs, record.rhs, record.result))
    return sentences


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def generate_dataset(
    n_equations: int = 200,
    *,
    max_int: int = 20,
    max_den: int = 8,
    seed: int = 0,
    holdout_frac: float = 0.1,
    split_seed: int = 1337,
    output: Path | None = None,
) -> Path:
    """Generate corpus files under ``datasets/small/``."""
    if n_equations <= 0:
        raise ValueError(f"n_equations must be positive, got {n_equations}")

    all_equations = iter_valid_equations(max_int=max_int, max_den=max_den)
    if not all_equations:
        raise RuntimeError("no valid equations generated; widen max_int or max_den")

    rng = random.Random(seed)
    rng.shuffle(all_equations)
    selected = all_equations[: min(n_equations, len(all_equations))]

    root = Path(output) if output is not None else default_dataset_dir()
    root.mkdir(parents=True, exist_ok=True)

    train_records, holdout_records = split_equations(
        selected,
        holdout_frac=holdout_frac,
        seed=split_seed,
    )

    train_json = [record.to_json() for record in train_records]
    holdout_json = [record.to_json() for record in holdout_records]
    all_json = train_json + holdout_json

    train_sentences = records_to_sentences(train_records)
    all_sentences = records_to_sentences(selected)

    (root / "corpus_train.txt").write_text(
        "\n".join(train_sentences) + ("\n" if train_sentences else ""),
        encoding="utf-8",
    )
    (root / "corpus.txt").write_text(
        "\n".join(all_sentences) + ("\n" if all_sentences else ""),
        encoding="utf-8",
    )

    write_jsonl(root / "equations_train.jsonl", train_json)
    write_jsonl(root / "equations_holdout.jsonl", holdout_json)
    write_jsonl(root / "equations.jsonl", all_json)

    manifest = {
        "dataset": DEFAULT_DATASET_NAME,
        "n_equations": len(selected),
        "n_train_equations": len(train_records),
        "n_holdout_equations": len(holdout_records),
        "n_sentences": len(all_sentences),
        "n_train_sentences": len(train_sentences),
        "max_int": max_int,
        "max_den": max_den,
        "seed": seed,
        "split_seed": split_seed,
        "holdout_frac": holdout_frac,
        "ops": list(OPS),
        "domains": ["I", "Q"],
        "available_equations": len(all_equations),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def load_corpus(root: Path | str | None = None) -> str:
    """Load generated ``corpus.txt`` as a single string."""
    root = Path(root) if root is not None else default_dataset_dir()
    path = root / "corpus.txt"
    if not path.is_file():
        raise FileNotFoundError(f"corpus not found at {path}")
    return path.read_text(encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"equations not found at {path}")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_equations(root: Path | str | None = None) -> list[dict]:
    root = Path(root) if root is not None else default_dataset_dir()
    return _load_jsonl(root / "equations.jsonl")


def load_equations_split(
    root: Path | str | None = None,
) -> tuple[list[dict], list[dict]]:
    root = Path(root) if root is not None else default_dataset_dir()
    train_path = root / "equations_train.jsonl"
    holdout_path = root / "equations_holdout.jsonl"
    if train_path.is_file() and holdout_path.is_file():
        return _load_jsonl(train_path), _load_jsonl(holdout_path)
    rows = load_equations(root)
    train_rows, holdout_rows = [], []
    for row in rows:
        key = equation_key(row)
        if hash(key) % 10 == 0:
            holdout_rows.append(row)
        else:
            train_rows.append(row)
    if not holdout_rows and rows:
        holdout_rows = [rows[0]]
        train_rows = rows[1:]
    return train_rows, holdout_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Small English math lexical corpus generator")
    parser.add_argument("--generate", action="store_true", help="write datasets/small corpus files")
    parser.add_argument("--n-equations", type=int, default=200)
    parser.add_argument("--max-int", type=int, default=20)
    parser.add_argument("--max-den", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.generate:
        root = generate_dataset(
            n_equations=args.n_equations,
            max_int=args.max_int,
            max_den=args.max_den,
            seed=args.seed,
            holdout_frac=args.holdout_frac,
            split_seed=args.split_seed,
            output=args.output,
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        print(f"Saved {manifest['n_equations']} equations ({manifest['n_sentences']} sentences) to {root}")
        sample = load_equations(root)[0]["sentences"][0]
        print(f"  sample: {sample}")
    else:
        parser.print_help()
