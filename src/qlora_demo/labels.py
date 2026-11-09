from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

IGNORE_INDEX = -100


class Tokenizer(Protocol):
    def __call__(self, text: str, add_special_tokens: bool = ...) -> dict[str, list[int]]: ...


class LabelTruncationError(ValueError):
    pass


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def naive_encode(tokenizer: Tokenizer, prompt: str, completion: str, max_length: int) -> EncodedExample:
    """The common buggy recipe: tokenize prompt+completion, truncate from the right, then mask the prompt."""
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"][:max_length]
    labels = [IGNORE_INDEX] * min(prompt_len, len(ids)) + ids[prompt_len:]
    return EncodedExample(input_ids=ids, labels=labels)


def validate_labels(example: EncodedExample, eos_token_id: int, min_supervised: int = 1) -> None:
    if len(example.input_ids) != len(example.labels):
        raise LabelTruncationError("input_ids and labels have different lengths")
    if example.supervised_tokens < min_supervised:
        raise LabelTruncationError(
            f"only {example.supervised_tokens} supervised tokens (min {min_supervised}); completion was truncated away"
        )
    if example.labels[-1] != eos_token_id:
        raise LabelTruncationError("last supervised token is not EOS; completion was cut before its end")


def encode_prompt_completion(
    tokenizer: Tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    eos_token_id: int,
    min_supervised: int = 8,
) -> EncodedExample | None:
    """Tokenize prompt and completion separately and drop the example instead of truncating it."""
    eos = eos_token_id
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids or completion_ids[-1] != eos:
        completion_ids = [*completion_ids, eos]
    if len(prompt_ids) + len(completion_ids) > max_length:
        return None
    example = EncodedExample(
        input_ids=prompt_ids + completion_ids,
        labels=[IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    )
    validate_labels(example, eos, min_supervised)
    return example


def audit(examples: Sequence[EncodedExample], eos_token_id: int) -> dict[str, int]:
    broken = 0
    fully_masked = 0
    for example in examples:
        if example.supervised_tokens == 0:
            fully_masked += 1
        try:
            validate_labels(example, eos_token_id)
        except LabelTruncationError:
            broken += 1
    return {"total": len(examples), "broken": broken, "fully_masked": fully_masked}
