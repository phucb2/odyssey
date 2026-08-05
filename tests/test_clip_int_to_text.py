"""Tests for clip.int_to_text — fixed cases plus random checks vs a reference."""
from __future__ import annotations

import random

import pytest

from odyssey.experiments.clip import int_to_text


def _reference_int_to_text(n: int) -> str:
    """Independent English spelling (and only after hundreds), for cross-checks."""
    if n == 0:
        return "zero"
    if n < 0:
        return f"minus {_reference_int_to_text(-n)}"

    small = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]
    decade = {
        20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
        60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    }
    units = ("", "thousand", "million", "billion", "trillion")

    def spell_0_999(x: int) -> str:
        if x == 0:
            return ""
        h, r = divmod(x, 100)
        bits: list[str] = []
        if h:
            bits.append(f"{small[h]} hundred")
            if r:
                bits.append("and")
        if r == 0:
            return " ".join(bits)
        if r < 20:
            bits.append(small[r])
        else:
            d = (r // 10) * 10
            bits.append(decade[d] if r == d else f"{decade[d]} {small[r - d]}")
        return " ".join(bits)

    out: list[str] = []
    for i, name in enumerate(units):
        n, chunk = divmod(n, 1000)
        if chunk:
            piece = spell_0_999(chunk)
            out.append(f"{piece} {name}".strip() if name else piece)
        if n == 0:
            break
    else:
        if n:
            raise ValueError("n too large")
    return " ".join(reversed(out))


FIXED_CASES = {
    0: "zero",
    10: "ten",
    12: "twelve",
    20: "twenty",
    21: "twenty one",
    99: "ninety nine",
    100: "one hundred",
    101: "one hundred and one",
    112: "one hundred and twelve",
    200: "two hundred",
    999: "nine hundred and ninety nine",
    1000: "one thousand",
    1100: "one thousand one hundred",
    1101: "one thousand one hundred and one",
    9999: "nine thousand nine hundred and ninety nine",
    -7: "minus seven",
}


@pytest.mark.parametrize("n, expected", FIXED_CASES.items())
def test_int_to_text_fixed(n: int, expected: str) -> None:
    assert int_to_text(n) == expected


@pytest.mark.parametrize("seed", range(5))
def test_int_to_text_random(seed: int) -> None:
    rng = random.Random(seed)
    values = [rng.randint(-10_000, 10_000) for _ in range(50)]
    values += [rng.randint(0, 10**12 - 1) for _ in range(20)]
    for n in values:
        got = int_to_text(n)
        want = _reference_int_to_text(n)
        assert got == want, f"n={n}: got={got!r} want={want!r}"
