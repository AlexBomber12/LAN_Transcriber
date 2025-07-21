import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lan_transcriber import normalizer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello hello hello", "hello"),
        ("Ciao mondo. Ciao mondo!", "Ciao mondo."),
        ("different sentences.", "different sentences."),
        ("hi", ""),
    ],
)
def test_dedup_cases(text: str, expected: str) -> None:
    assert normalizer.dedup(text) == expected


def test_whitespace_punctuation() -> None:
    assert normalizer.dedup(" Hello!  Hello!  ") == "Hello!"
