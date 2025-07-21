import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_dedup_simple():
    from lan_transcriber import normalizer

    assert normalizer.dedup("hi hi hi") == "hi"


def test_dedup_mixed_language():
    from lan_transcriber import normalizer

    assert normalizer.dedup("Ciao mondo. Ciao mondo!") == "Ciao mondo."


def test_dedup_preserve():
    from lan_transcriber import normalizer

    text = "Hello world. How are you?"
    assert normalizer.dedup(text) == text


def test_whitespace_punctuation():
    from lan_transcriber import normalizer

    assert normalizer.dedup(" Hello!  Hello!  ") == "Hello!"
