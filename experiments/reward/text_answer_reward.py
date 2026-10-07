"""Exact numeric answer scoring with limited presentation normalization.

Accepted labels are FINAL_ANSWER: and Final Answer: (case-insensitive).
The structured FINAL_ANSWER: label is authoritative whenever present; earlier
natural-language answer prose is ignored. Final Answer: is a fallback only
when no structured label occurs. The selected answer must occupy the final
meaningful line. Earlier labels of that selected kind must be empty
decorations, and repeated empty labels on the final line are allowed.
Whole-line bold/math wrappers and flat LaTeX text wrappers may be removed;
markup inside a number is never removed. Standalone display-math delimiter
lines are ignored. Empty Markdown answer headings are decorations only.
This module intentionally leaves the strict v2 verifier unchanged.
"""

from fractions import Fraction
import re

from text_reward import _parse_number, canonical


_LABEL_TEXT = r"(?:FINAL_ANSWER|Final[ \t]+Answer)"
_LABEL = re.compile(rf"{_LABEL_TEXT}[ \t]*:", re.IGNORECASE | re.ASCII)
_PRIMARY = re.compile(r"FINAL_ANSWER[ \t]*:", re.IGNORECASE | re.ASCII)
_BOLD_LABEL = re.compile(rf"^\*\*({_LABEL_TEXT})\*\*([ \t]*:)", re.IGNORECASE | re.ASCII)
_EMPTY_HEADING = re.compile(rf"^\#{{1,6}}[ \t]+({_LABEL_TEXT}[ \t]*:)[ \t]*$", re.IGNORECASE | re.ASCII)
_TEXT = re.compile(r"\\text\{([^{}]*)\}(.*)", re.ASCII)
_DISPLAY_DELIMITERS = {"$$", r"\[", r"\]"}


def _normalize_line(line: str) -> str:
    line = line.strip()
    for _ in range(4):
        previous = line
        for wrapper in ("**", "$$", "$"):
            if line.startswith(wrapper) and line.endswith(wrapper) and len(line) > 2 * len(wrapper):
                inner = line[len(wrapper):-len(wrapper)]
                # A wrapper must be balanced around the entire line.
                if wrapper not in inner and (wrapper != "$" or "$" not in inner):
                    line = inner.strip()
                    break
        match = _TEXT.fullmatch(line)
        if match:
            contents, suffix = match.groups()
            # Unwrap only an answer label/value, never arbitrary LaTeX.
            if _LABEL.match(contents.strip()):
                if not suffix.strip() or _LABEL.fullmatch(contents.strip()):
                    line = (contents.strip() + " " + suffix.strip()).strip()
        line = _BOLD_LABEL.sub(r"\1\2", line)
        heading = _EMPTY_HEADING.fullmatch(line)
        if heading:
            line = heading.group(1)
        if line == previous:
            break
    return line


def parse_answer(text: str) -> Fraction | None:
    """Return one terminal exact numeric answer, or None for invalid text."""
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [_normalize_line(line) for line in lines if line not in _DISPLAY_DELIMITERS]
    selected_label = _PRIMARY if any(_PRIMARY.search(line) for line in lines) else _LABEL
    saw_empty_label = False
    for index, line in enumerate(lines):
        if selected_label.search(line) is None:
            if saw_empty_label:
                return None  # A label cannot be followed by unrelated prose.
            continue
        remaining = line
        labels = 0
        while match := _LABEL.match(remaining):
            labels += 1
            remaining = remaining[match.end():].strip()
        if not labels:
            return None  # Embedded labels after nonempty prose are ambiguous.
        if not remaining:
            saw_empty_label = True
            continue
        # Full numeric parsing rejects a value before another label, units,
        # expressions, malformed markup, and alternative numeric answers.
        value = _parse_number(remaining)
        if value is None or index != len(lines) - 1:
            return None
        return value
    return None


def score_answer(text: str, gold_value: str) -> dict:
    expected = _parse_number(gold_value)
    if expected is None:
        raise ValueError("gold_value must be a valid bounded numeric string.")
    prediction = parse_answer(text)
    return {
        "prediction": canonical(prediction) if prediction is not None else None,
        "answer_valid": prediction is not None,
        "correct": prediction is not None and prediction == expected,
    }


if __name__ == "__main__":
    import unittest

    class AnswerRewardTests(unittest.TestCase):
        def test_observed_correct_presentations(self):
            cases = {
                "Final answer: FINAL_ANSWER: 4000": "4000",  # student 1373
                "$$\n\\text{FINAL_ANSWER: } 250\n$$": "250",  # student 1327
                "**Final Answer**: FINAL_ANSWER: 19": "19",  # student 3454
                "**FINAL_ANSWER: 950**": "950",  # teacher 3045
                "Final Answer: 15": "15",  # teacher 4881
                "**FINAL_ANSWER: 52**": "52",  # teacher 3423
                "### Final Answer:\n$$\n\\text{FINAL_ANSWER: 4000}\n$$": "4000",  # teacher 3142
            }
            for text, expected in cases.items():
                with self.subTest(text=text):
                    self.assertTrue(score_answer("Reasoning: 999.\n" + text, expected)["correct"])

        def test_observed_wrong_numbers_stay_wrong(self):
            for text, gold in (("Final answer: FINAL_ANSWER: 60", "20"),
                               ("**FINAL_ANSWER: 23.75**", "25"),
                               ("**FINAL_ANSWER: 840**", "560")):
                with self.subTest(text=text):
                    result = score_answer(text, gold)
                    self.assertTrue(result["answer_valid"])
                    self.assertFalse(result["correct"])
            self.assertIsNone(parse_answer("Final Answer:\nFINAL_ANSWER: 120 counters and 200 marbles"))

        def test_allowed_numeric_equivalence(self):
            for text in ("FINAL_ANSWER:1/2", "final_answer: .5", "Final Answer: 0.50",
                         "**$FINAL_ANSWER: 2/4$**", "$$FINAL_ANSWER: 1/2$$",
                         "\\[\nFINAL_ANSWER: 0.5\n\\]", "\\text{Final Answer:} 1/2"):
                with self.subTest(text=text):
                    self.assertEqual(parse_answer(text), Fraction(1, 2))
            self.assertEqual(parse_answer("FINAL_ANSWER: \u22121,234.5"), Fraction(-2469, 2))

        def test_structured_label_takes_precedence_over_earlier_prose(self):
            cases = ["Final Answer:41\nFINAL_ANSWER:42",
                     "Final Answer:42\nFINAL_ANSWER:42",
                     "Final Answer: 502 dollars\nFINAL_ANSWER:502",
                     "The Final Answer: 502.\nSome explanation.\nFINAL_ANSWER:502",
                     "Final Answer:\n$502$\nFINAL_ANSWER:502"]
            for text in cases:
                with self.subTest(text=text):
                    expected = Fraction(42 if text.endswith("42") else 502)
                    self.assertEqual(parse_answer(text), expected)
            self.assertIsNone(parse_answer("FINAL_ANSWER:41\nFINAL_ANSWER:42"))
            self.assertIsNone(parse_answer("FINAL_ANSWER:42\nFINAL_ANSWER:42"))
            self.assertIsNone(parse_answer("FINAL_ANSWER:42\nFinal Answer:42"))
            self.assertIsNone(parse_answer("Final Answer:41\nFinal Answer:42"))

        def test_rejects_ambiguous_or_malformed_answers(self):
            cases = ["42", "#### 42", "FINAL_ANSWER:", "FINAL_ANSWER:42\nTrailing text",
                     "FINAL_ANSWER:41\nFINAL_ANSWER:42", "FINAL_ANSWER:42\nFINAL_ANSWER:42",
                     "FINAL_ANSWER:42 FINAL_ANSWER:42", "Final Answer: wrong FINAL_ANSWER:42",
                     "The Final Answer:42", "FINAL_ANSWER:42\nFinal Answer:",
                     "FINAL_ANSWER:\nUnrelated text\nFINAL_ANSWER:42",
                     "FINAL_ANSWER:4**2", "FINAL_ANSWER:4$2", "FINAL_ANSWER:4\\text{}2",
                     "FINAL_ANSWER:4*2", "FINAL_ANSWER:4+2", "FINAL_ANSWER:42 or 43",
                     "FINAL_ANSWER:$42", "FINAL_ANSWER:42 dollars", "FINAL_ANSWER:42%",
                     "FINAL_ANSWER:1,00", "FINAL_ANSWER:NaN", "FINAL_ANSWER:Infinity",
                     "FINAL_ANSWER:1/0", "FINAL_ANSWER:--42", "FINAL_ANSWER:42**",
                     "**FINAL_ANSWER:42", "$FINAL_ANSWER:42$$",
                     r"\text{Other label:} FINAL_ANSWER:42", r"\text{FINAL_ANSWER:4}2",
                     r"\text{FINAL_ANSWER:{42}}", "FINAL_ANSWER:42\n```"]
            for text in cases:
                with self.subTest(text=text):
                    self.assertIsNone(parse_answer(text))

        def test_wrong_sign_exactness_and_bounds(self):
            self.assertFalse(score_answer("FINAL_ANSWER:-42", "42")["correct"])
            self.assertFalse(score_answer("FINAL_ANSWER:0.3333333333333333", "1/3")["correct"])
            self.assertIsNone(parse_answer("FINAL_ANSWER:" + "9" * 129))
            self.assertIsNotNone(parse_answer("FINAL_ANSWER:" + "9" * 128))
            self.assertEqual(score_answer("No answer", "42"),
                             {"prediction": None, "answer_valid": False, "correct": False})
            with self.assertRaises(ValueError):
                score_answer("FINAL_ANSWER:42", "1/0")

    unittest.main(verbosity=2)
