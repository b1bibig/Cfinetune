"""평가·학습 프롬프트. 채팅 형식은 모델 구조별 어댑터(arch.py)가 정한다."""

from __future__ import annotations

MCQ_INSTRUCTION = {
    "ko": "다음 문제의 정답을 고르세요. 답은 보기의 알파벳 하나로만 하세요.",
    "en": "Choose the correct answer. Reply with a single letter.",
}
ANSWER_PREFIX = {"ko": "정답:", "en": "Answer:"}


def mcq_body(item: dict) -> str:
    opts = "\n".join(f"{'ABCDE'[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n{opts}"


def mcq_prompt(arch, item: dict, fmt: str = "chat") -> str:
    """답변 자리에 '정답:'까지 미리 써 두고, 다음 토큰(알파벳)의 확률을 본다."""
    lang = item["lang"]
    text = f"{MCQ_INSTRUCTION[lang]}\n\n{mcq_body(item)}"
    if fmt == "chat":
        return arch.chat_prefix(text) + ANSWER_PREFIX[lang]
    if fmt == "plain":
        return f"{text}\n{ANSWER_PREFIX[lang]}"
    raise ValueError(fmt)


def chat_prompt(arch, user: str) -> str:
    return arch.chat_prefix(user)


def sft_text(arch, prompt: str, response: str) -> tuple[str, str]:
    """(손실을 빼는 앞부분, 손실을 계산할 답변 부분)."""
    return arch.chat_prefix(prompt), response + arch.END
