from __future__ import annotations

import re

SYSTEM_PROMPT = "You are a careful math tutor."
INSTRUCTION = (
    "Solve the following grade-school math problem. Think step by step, "
    "then finish with a final line of the form 'The answer is: <number>'."
)

_CALCULATOR = re.compile(r"<<[^>]*>>")
_FINAL = re.compile(r"\n?####\s*(.+)$")


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\nProblem: {question}"},
    ]


def format_solution(solution: str) -> str:
    text = _CALCULATOR.sub("", solution.strip())
    match = _FINAL.search(text)
    if not match:
        raise ValueError("GSM8K solution without '####' marker")
    answer = match.group(1).strip().replace(",", "")
    body = text[: match.start()].rstrip()
    return f"{body}\nThe answer is: {answer}"
