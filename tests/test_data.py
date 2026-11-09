import pytest

from qlora_demo import data
from qlora_demo.data import build_messages, format_solution


def test_format_solution_strips_calculator_and_rewrites_final_line() -> None:
    raw = "She sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\nShe makes 9 * 2 = $<<9*2=18>>18.\n#### 1,018"
    assert format_solution(raw) == "She sells 16 - 3 - 4 = 9 eggs.\nShe makes 9 * 2 = $18.\nThe answer is: 1018"


def test_format_solution_requires_marker() -> None:
    with pytest.raises(ValueError):
        format_solution("no final answer")


def test_messages_shape() -> None:
    messages = build_messages("What is 2+2?")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].endswith("Problem: What is 2+2?")


def test_prompt_matches_eval_harness_when_installed() -> None:
    tasks = pytest.importorskip("evalh.tasks")
    assert data.SYSTEM_PROMPT == tasks.SYSTEM_PROMPT
    assert data.INSTRUCTION == tasks.INSTRUCTION
    assert build_messages("x") == tasks.build_messages("x")
