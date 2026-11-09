import pytest

from qlora_demo.labels import (
    IGNORE_INDEX,
    EncodedExample,
    LabelTruncationError,
    audit,
    encode_prompt_completion,
    naive_encode,
    validate_labels,
)

EOS_TEXT = "|"
EOS = ord(EOS_TEXT)


class CharTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


TOK = CharTokenizer()
SHORT_PROMPT = "Q: 2+2? A: "
LONG_PROMPT = "Q: " + "very long context " * 10 + "A: "
COMPLETION = "2+2 is 4. The answer is: 4" + EOS_TEXT


def test_naive_recipe_masks_everything_when_prompt_fills_the_window() -> None:
    example = naive_encode(TOK, LONG_PROMPT, COMPLETION, max_length=64)
    assert example.supervised_tokens == 0
    with pytest.raises(LabelTruncationError, match="truncated away"):
        validate_labels(example, EOS)


def test_naive_recipe_drops_eos_when_completion_is_cut() -> None:
    example = naive_encode(TOK, SHORT_PROMPT, COMPLETION, max_length=len(SHORT_PROMPT) + 10)
    assert example.supervised_tokens == 10
    with pytest.raises(LabelTruncationError, match="not EOS"):
        validate_labels(example, EOS)


def test_guarded_encoder_drops_overlong_examples() -> None:
    assert encode_prompt_completion(TOK, LONG_PROMPT, COMPLETION, max_length=64, eos_token_id=EOS) is None


def test_guarded_encoder_keeps_full_completion_and_eos() -> None:
    example = encode_prompt_completion(TOK, SHORT_PROMPT, COMPLETION, max_length=128, eos_token_id=EOS)
    assert example is not None
    assert example.labels[: len(SHORT_PROMPT)] == [IGNORE_INDEX] * len(SHORT_PROMPT)
    assert example.labels[len(SHORT_PROMPT) :] == [ord(c) for c in COMPLETION]
    assert example.labels[-1] == EOS


def test_guarded_encoder_appends_missing_eos() -> None:
    example = encode_prompt_completion(TOK, SHORT_PROMPT, "The answer is: 4", max_length=128, eos_token_id=EOS)
    assert example is not None
    assert example.input_ids[-1] == EOS


def test_validate_rejects_length_mismatch() -> None:
    with pytest.raises(LabelTruncationError):
        validate_labels(EncodedExample(input_ids=[1, 2], labels=[EOS]), EOS)


def test_audit_counts_broken_examples() -> None:
    examples = [
        naive_encode(TOK, LONG_PROMPT, COMPLETION, 64),
        naive_encode(TOK, SHORT_PROMPT, COMPLETION, len(SHORT_PROMPT) + 10),
        naive_encode(TOK, SHORT_PROMPT, COMPLETION, 128),
    ]
    assert audit(examples, EOS) == {"total": 3, "broken": 2, "fully_masked": 1}
