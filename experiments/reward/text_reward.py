"""Exact numeric reward for the text RL/OPD suitability pilot.

Completion grammar: exactly one ``FINAL_ANSWER:`` marker in the entire
completion, on its own final nonempty line, followed by whitespace and one
number. Dataset gold solutions retain their separate ``####`` delimiter.
Numbers are signed integers, decimals (including .5 and 5.), or integer
fractions such as -1/2. Commas must group thousands correctly. A fraction
may have signed numerator/denominator but no spaces around its slash.
Unicode minus (U+2212) is explicitly normalized to ASCII minus.

The verifier does not execute expressions, use float tolerances, or infer
an answer from other numbers in the completion. Generated special tokens
must be removed by the caller before scoring. This is a custom evaluator,
not an exact reproduction of the original GSM8K first-match regex scorer.
"""

from fractions import Fraction
import re


MAX_NUMERIC_CHARS = 128
_INTEGER = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_DECIMAL = rf"(?:{_INTEGER}(?:\.[0-9]*)?|\.[0-9]+)"
_NUMBER = re.compile(rf"[+-]?(?:{_DECIMAL}|{_INTEGER}/[+-]?{_INTEGER})", re.ASCII)
_FINAL_LINES = {
    marker: re.compile(rf"[ \t]*{re.escape(marker)}[ \t]+(.+?)[ \t]*", re.ASCII)
    for marker in ("FINAL_ANSWER:", "####")
}


def _parse_number(payload: str) -> Fraction | None:
    """Parse a bounded numeric payload under the documented grammar."""
    if not isinstance(payload, str) or not 1 <= len(payload) <= MAX_NUMERIC_CHARS:
        return None
    normalized = payload.replace("\u2212", "-")
    if _NUMBER.fullmatch(normalized) is None:
        return None
    normalized = normalized.replace(",", "")
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/")
            return Fraction(int(numerator), int(denominator))
        return Fraction(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_marked_final(text: str, marker: str) -> Fraction | None:
    """Parse one explicitly selected delimiter without fallback matching."""
    if not isinstance(text, str) or text.count(marker) != 1:
        return None
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = _FINAL_LINES[marker].fullmatch(lines[-1])
    if match is None:
        return None
    return _parse_number(match.group(1))


def parse_final(text: str) -> Fraction | None:
    """Parse a FINAL_ANSWER completion; invalid format returns None."""
    return _parse_marked_final(text, "FINAL_ANSWER:")


def parse_gold(answer: str) -> Fraction:
    """Parse a full dataset solution; malformed answer keys are errors."""
    value = _parse_marked_final(answer, "####")
    if value is None:
        raise ValueError("Gold solution lacks exactly one valid final #### number line.")
    return value


def canonical(value: Fraction) -> str:
    """Serialize an exact value as an integer or reduced fraction."""
    if not isinstance(value, Fraction):
        raise TypeError("canonical expects fractions.Fraction, not a float.")
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def score_completion(text: str, gold_value: str) -> dict:
    """Score generated text against a numeric key, not a full gold solution.

    A malformed gold value raises ValueError rather than silently assigning
    incorrect rewards to every response.
    """
    expected = _parse_number(gold_value)
    if expected is None:
        raise ValueError("gold_value must be a valid numeric string, such as '42' or '1/2'.")
    prediction = parse_final(text)
    return {
        "prediction": canonical(prediction) if prediction is not None else None,
        "format_valid": prediction is not None,
        "correct": prediction is not None and prediction == expected,
    }


if __name__ == "__main__":
    import unittest

    class RewardVerifierTests(unittest.TestCase):
        def test_equivalent_numbers_and_signs(self):
            cases = {
                "42": Fraction(42),
                "+42": Fraction(42),
                "42.0": Fraction(42),
                "42.": Fraction(42),
                "00042": Fraction(42),
                "-3": Fraction(-3),
                "\u22123": Fraction(-3),
                "-0.0": Fraction(0),
                ".5": Fraction(1, 2),
                "-.5": Fraction(-1, 2),
                "0.50": Fraction(1, 2),
                "1/2": Fraction(1, 2),
                "2/4": Fraction(1, 2),
                "-1/-2": Fraction(1, 2),
                "1/\u22122": Fraction(-1, 2),
                "1,234": Fraction(1234),
                "-1,234.50": Fraction(-2469, 2),
                "1,000/2,000": Fraction(1, 2),
            }
            for payload, expected in cases.items():
                with self.subTest(payload=payload):
                    self.assertEqual(parse_final(f"Reasoning uses 999.\nFINAL_ANSWER: {payload}"), expected)

        def test_final_line_whitespace(self):
            self.assertEqual(parse_final("Work\r\n\tFINAL_ANSWER:\t42 \t\r\n\r\n"), Fraction(42))

        def test_malformed_or_ambiguous_answers(self):
            cases = [
                "", "The answer is 42", "Work FINAL_ANSWER: 42", "FINAL_ANSWER:42",
                "### FINAL_ANSWER: 42", "**FINAL_ANSWER:** 42", "final_answer: 42",
                "FINAL_ANSWER: ", "FINAL_ANSWER: 42\nMore text",
                "FINAL_ANSWER: 42\nFINAL_ANSWER: 42", "FINAL_ANSWER: 41\nFINAL_ANSWER: 42",
                "Mention FINAL_ANSWER: here\nFINAL_ANSWER: 42", "FINAL_ANSWER: 42 or 43",
                "FINAL_ANSWER: 42, 43", "FINAL_ANSWER: $42", "FINAL_ANSWER: 42 dollars",
                "FINAL_ANSWER: 50%", "FINAL_ANSWER: 1+1", "FINAL_ANSWER: 2*3",
                "FINAL_ANSWER: 2**3", "FINAL_ANSWER: 2=2", "FINAL_ANSWER: \\boxed{42}",
                "FINAL_ANSWER: 1e3", "FINAL_ANSWER: NaN", "FINAL_ANSWER: Infinity",
                "FINAL_ANSWER: inf", "FINAL_ANSWER: 1/0", "FINAL_ANSWER: 0/0",
                "FINAL_ANSWER: 1/2/3", "FINAL_ANSWER: 1 / 2", "FINAL_ANSWER: 0.5/2",
                "FINAL_ANSWER: --1", "FINAL_ANSWER: +-1", "FINAL_ANSWER: 1,00",
                "FINAL_ANSWER: 12,34", "FINAL_ANSWER: 1234,567", "FINAL_ANSWER: 1,,000",
                "FINAL_ANSWER: ,100", "FINAL_ANSWER: 1,234,", "FINAL_ANSWER: .",
                "FINAL_ANSWER: 1.2.3", "FINAL_ANSWER: 1 000", "FINAL_ANSWER: \u0664\u0662",
                "FINAL_ANSWER: 42\n```", "FINAL_ANSWER: __import__('os').system('echo bad')",
            ]
            for completion in cases:
                with self.subTest(completion=completion):
                    self.assertIsNone(parse_final(completion))

        def test_payload_length_bound(self):
            self.assertIsNotNone(parse_final("FINAL_ANSWER: " + "9" * MAX_NUMERIC_CHARS))
            self.assertIsNone(parse_final("FINAL_ANSWER: " + "9" * (MAX_NUMERIC_CHARS + 1)))

        def test_exact_arithmetic_without_tolerance(self):
            self.assertTrue(score_completion("FINAL_ANSWER: 0.5", "1/2")["correct"])
            self.assertTrue(score_completion("FINAL_ANSWER: 42.0", "42")["correct"])
            self.assertTrue(score_completion("FINAL_ANSWER: -0", "0")["correct"])
            self.assertFalse(score_completion("FINAL_ANSWER: 0.3333333333333333", "1/3")["correct"])
            self.assertFalse(score_completion("FINAL_ANSWER: 1.9999999999999999", "2")["correct"])

        def test_scores_separate_format_and_correctness(self):
            self.assertEqual(score_completion("FINAL_ANSWER: 2/4", "1/2"),
                             {"prediction": "1/2", "format_valid": True, "correct": True})
            self.assertEqual(score_completion("FINAL_ANSWER: 43", "42"),
                             {"prediction": "43", "format_valid": True, "correct": False})
            self.assertEqual(score_completion("Answer: 42", "42"),
                             {"prediction": None, "format_valid": False, "correct": False})

        def test_gold_is_checked_and_full_solution_is_not_numeric_key(self):
            self.assertEqual(parse_gold("Compute <<6*7=42>>42.\n#### 42"), Fraction(42))
            self.assertEqual(parse_gold("Work\n#### 0.50"), Fraction(1, 2))
            self.assertEqual(parse_gold("Work\n#### 1,234"), Fraction(1234))
            for bad_gold in ("42", "#### 42\n#### 43", "#### 1/0", "Work #### 42",
                             "####42", "#### 42\nMore text", "Mention ####\n#### 42"):
                with self.subTest(bad_gold=bad_gold), self.assertRaises(ValueError):
                    parse_gold(bad_gold)
            for bad_key in ("#### 42", "FINAL_ANSWER: 42", "1/0", "NaN", "not a number"):
                with self.subTest(bad_key=bad_key), self.assertRaises(ValueError):
                    score_completion("FINAL_ANSWER: 42", bad_key)

        def test_completion_and_gold_delimiters_are_distinct(self):
            self.assertIsNone(parse_final("Work\n#### 42"))
            self.assertFalse(score_completion("Work\n#### 42", "42")["correct"])
            with self.assertRaises(ValueError):
                parse_gold("Work\nFINAL_ANSWER: 42")

        def test_canonical_form(self):
            self.assertEqual(canonical(Fraction(4, 2)), "2")
            self.assertEqual(canonical(Fraction(-2, 4)), "-1/2")
            self.assertEqual(canonical(Fraction(0, 9)), "0")
            with self.assertRaises(TypeError):
                canonical(0.5)

    unittest.main(verbosity=2)
