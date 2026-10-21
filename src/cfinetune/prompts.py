"""gpt-oss harmony 형식 프롬프트.

transformers 채팅 템플릿은 "Current date"에 오늘 날짜를 넣어서 날마다 프롬프트가 달라진다.
실험 재현성을 위해 같은 형식을 날짜만 고정해서 직접 만든다.
"""

from __future__ import annotations

FIXED_DATE = "2026-01-01"
FINAL = "<|channel|>final<|message|>"
RETURN = "<|return|>"

_SYSTEM = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
           "Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: {reasoning}\n\n"
           "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")

MCQ_INSTRUCTION = {
    "ko": "다음 문제의 정답을 고르세요. 답은 보기의 알파벳 하나로만 하세요.",
    "en": "Choose the correct answer. Reply with a single letter.",
}
ANSWER_PREFIX = {"ko": "정답:", "en": "Answer:"}


def harmony_prefix(user: str, reasoning: str = "low", date: str = FIXED_DATE) -> str:
    """assistant 차례 직전까지 (끝이 <|start|>assistant)."""
    return _SYSTEM.format(date=date, reasoning=reasoning) + f"<|start|>user<|message|>{user}<|end|><|start|>assistant"


def mcq_body(item: dict) -> str:
    letters = "ABCDE"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n{opts}"


def mcq_prompt(item: dict, fmt: str = "harmony") -> str:
    lang = item["lang"]
    text = f"{MCQ_INSTRUCTION[lang]}\n\n{mcq_body(item)}"
    if fmt == "harmony":
        # 분석(analysis) 채널을 건너뛰고 최종 답 채널을 미리 열어 둔 뒤 다음 토큰(알파벳)의 확률을 본다
        return harmony_prefix(text) + FINAL + ANSWER_PREFIX[lang]
    if fmt == "plain":
        return f"{text}\n{ANSWER_PREFIX[lang]}"
    raise ValueError(fmt)


def chat_prompt(user: str) -> str:
    return harmony_prefix(user) + FINAL


def sft_text(prompt: str, response: str) -> tuple[str, str]:
    """(학습에서 손실을 빼는 앞부분, 손실을 계산할 뒷부분)."""
    return harmony_prefix(prompt), FINAL + response + RETURN
