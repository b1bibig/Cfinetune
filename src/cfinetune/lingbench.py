"""한국어 언어력 벤치: 문법·언어 지식 객관식, 번역투 최소쌍, 생성한 글의 말투 통계.

- 객관식: KoBALT-700(언어학 24현상, 보기 10개), KoGEM(문법 1524문항), CLIcK 언어 영역(650문항)
- 번역투 최소쌍(data/translationese_pairs.tsv): 같은 뜻의 자연스러운 문장과 번역투 문장 중 어느 쪽에 확률을 더 주나
- 생성 통계: 말투 지시(data/style_prompts.tsv) + K2-Eval 생성 문항(90개)에 답하게 한 뒤
  번역투 표지 빈도, 반말/존댓말 지시 준수율, 반복, 한글·한자·로마자 비율, 끝맺음(EOS) 비율을 센다.

토큰화와 채팅 형식은 llama-server(/tokenize, /apply-template)에 맡기므로 GGUF만 있으면 된다.
어휘가 다른 모델끼리도 같은 글자 단위로 비교한다 (번역투 점수는 글자당 로그확률).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

LETTERS = "ABCDEFGHIJ"
MCQ_INSTRUCTION = "다음 문제의 정답을 고르세요. 답은 보기의 알파벳 하나로만 하세요."
DATA = Path(__file__).resolve().parents[2] / "data"

HF = "https://huggingface.co/datasets"
CLICK_LANG = ["Functional/Functional_CSAT", "Functional/Functional_Kedu", "Functional/Functional_PSE",
              "Grammar/Grammar_CSAT", "Grammar/Grammar_Kedu", "Grammar/Grammar_TOPIK",
              "Textual/Textual_CSAT", "Textual/Textual_TOPIK"]


# ---------------------------------------------------------------- 데이터

def _get(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content.decode("utf-8")


def parse_kobalt(row: dict) -> dict:
    """KoBALT 문항은 지문·문제·보기(A: ~ J: ~)가 한 글에 들어 있다."""
    q = row["Question"]
    first = re.search(r"^[A-J]: ", q, re.M)
    choices = re.findall(r"^[A-J]: (.*)$", q[first.start():], re.M)
    return {"task": "kobalt", "subject": row["Class"], "question": q[:first.start()].strip(),
            "choices": choices, "answer": LETTERS.index(row["Answer"].strip())}


def load_kobalt() -> list[dict]:
    return [parse_kobalt(r) for r in json.loads(_get(f"{HF}/snunlp/KoBALT-700/resolve/main/data/train.jsonl"))]


def load_kogem() -> list[dict]:
    rows = json.loads(_get(f"{HF}/Poppo/KoGEM/resolve/main/dataset/KoGEM_benchmark.json"))
    out = []
    for r in rows:
        q = "\n\n".join(s for s in (r["passage"], r["paragraph"], r["question"]) if s and s.strip())
        out.append({"task": "kogem", "subject": r["level_1"], "question": q,
                    "choices": [re.sub(r"^\d\.\s*", "", c) for c in r["choices"]], "answer": r["label"] - 1})
    return out


def load_click_lang() -> list[dict]:
    out = []
    for f in CLICK_LANG:
        for r in json.loads(_get(f"{HF}/EunsuKim/CLIcK/resolve/main/Dataset/Language/{f}.json")):
            q = "\n\n".join(s for s in (r.get("paragraph", ""), r["question"]) if s and s.strip())
            out.append({"task": "click_lang", "subject": f.split("/")[0], "question": q,
                        "choices": r["choices"], "answer": r["choices"].index(r["answer"])})
    return out


def load_mcq() -> list[dict]:
    return load_kobalt() + load_kogem() + load_click_lang()


def _tsv(name: str) -> list[list[str]]:
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()
    return [l.split("\t") for l in lines if l.strip() and not l.startswith("#")]


def load_pairs() -> list[dict]:
    return [{"cat": c, "natural": n, "translationese": t} for c, n, t in _tsv("translationese_pairs.tsv")]


def load_gen_prompts() -> list[dict]:
    out = [{"set": "style", "register": reg, "prompt": p} for reg, p in _tsv("style_prompts.tsv")]
    k2 = csv.DictReader(io.StringIO(_get(f"{HF}/HAERAE-HUB/K2-Eval/resolve/main/data/k2-eval-generation.csv")))
    out += [{"set": "k2eval", "register": None, "prompt": r["instruction"]} for r in k2]
    return out


# ---------------------------------------------------------------- llama-server

class Client:
    def __init__(self, url: str):
        self.url = url

    def post(self, path: str, payload: dict) -> dict:
        r = requests.post(self.url + path, json=payload, timeout=900)
        r.raise_for_status()
        return r.json()

    def tokenize(self, text: str, special: bool = False, add_bos: bool = False) -> list[int]:
        return self.post("/tokenize", {"content": text, "parse_special": special, "add_special": add_bos})["tokens"]

    def chat_prompt(self, user: str) -> str:
        return self.post("/apply-template", {"messages": [{"role": "user", "content": user}]})["prompt"]


def letter_ids(cl: Client, n: int) -> list[set[int]]:
    """'A'와 ' A'처럼 알파벳 하나로 된 토큰."""
    out = []
    for L in LETTERS[:n]:
        out.append({t[0] for t in (cl.tokenize(L), cl.tokenize(" " + L)) if len(t) == 1})
    return out


def mcq_text(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{MCQ_INSTRUCTION}\n\n{item['question']}\n\n{opts}"


def score_mcq(cl: Client, items: list[dict], parallel: int = 4, progress=None) -> list[dict]:
    """채팅 형식으로 묻고 '정답:' 다음 토큰에서 보기 알파벳들의 확률을 비교한다."""
    letters = letter_ids(cl, len(LETTERS))

    def one(item):
        ids = cl.tokenize(cl.chat_prompt(mcq_text(item)) + "정답:", special=True)
        r = cl.post("/completion", {"prompt": ids, "n_predict": 1, "n_probs": 100, "temperature": -1, "cache_prompt": False})
        tops = {t["id"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
        scores = [max((tops.get(t, -math.inf) for t in letters[i]), default=-math.inf) for i in range(len(item["choices"]))]
        best = max(range(len(scores)), key=scores.__getitem__)
        pred = best if scores[best] > -math.inf else -1
        return {"task": item["task"], "subject": item["subject"], "answer": item["answer"], "pred": pred,
                "correct": pred == item["answer"], "n_choices": len(item["choices"])}

    out = []
    with ThreadPoolExecutor(parallel) as ex:
        for i, r in enumerate(ex.map(one, items)):
            out.append(r)
            if progress and i % 200 == 0:
                progress(i, len(items))
    return out


def summarize_mcq(res: list[dict]) -> dict:
    out = {}
    for task in sorted({r["task"] for r in res}):
        rs = [r for r in res if r["task"] == task]
        subj = {}
        for s in sorted({r["subject"] for r in rs}):
            ss = [r for r in rs if r["subject"] == s]
            subj[s] = round(sum(r["correct"] for r in ss) / len(ss), 4)
        out[task] = {"acc": sum(r["correct"] for r in rs) / len(rs), "n": len(rs),
                     "chance": sum(1 / r["n_choices"] for r in rs) / len(rs),
                     "no_letter": sum(r["pred"] < 0 for r in rs) / len(rs), "by_subject": subj}
    return out


def sentence_logprob(cl: Client, ctx: list[int], ids: list[int]) -> float:
    """ctx 다음에 ids가 나올 로그확률 (정확한 값). 토큰 id 문법으로 그 토큰들만 고르게 하고,
    고른 토큰마다 전체 어휘 소프트맥스 확률(temperature<0)을 받아 더한다. 요청 한 번."""
    def forced(prompt, toks):
        grammar = "root ::= " + " ".join(f"<[{t}]>" for t in toks)
        r = cl.post("/completion", {"prompt": prompt, "n_predict": len(toks), "temperature": -1, "grammar": grammar,
                                    "n_probs": 1, "return_tokens": True, "cache_prompt": True})
        if r.get("tokens") != toks:
            raise RuntimeError(f"강제한 토큰과 다르게 생성됨: {r.get('tokens')} != {toks}")
        return sum(c["logprob"] for c in r["completion_probabilities"])

    def one_token(prompt, t):
        """logit_bias로 t를 고르게 한다 (보고되는 확률은 bias 전 원래 값). 바이트 토큰은 글자가 완성 안 돼
        서버가 확률을 안 돌려주므로, 그 자리 분포 상위 1000개에서 찾는다 (없으면 1000번째 값 = 상한)."""
        r = cl.post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": -1, "logit_bias": [[t, 1000]],
                                    "n_probs": 1000, "return_tokens": True, "cache_prompt": True})
        probs = r.get("completion_probabilities")
        if probs:
            return probs[0]["logprob"]
        r = cl.post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": -1, "logit_bias": [[ctx[-1], 1000]],
                                    "n_probs": 1000, "cache_prompt": True})
        tops = r["completion_probabilities"][0]["top_logprobs"]
        return next((x["logprob"] for x in tops if x["id"] == t), tops[-1]["logprob"])

    try:
        return forced(ctx, ids)
    except requests.HTTPError:
        # 드문 글자가 바이트 토큰으로 쪼개지면 llama.cpp 문법 검사가 글자 중간에서 깨진다 -> 한 토큰씩
        return sum(one_token(ctx + ids[:i], t) for i, t in enumerate(ids))


def unigram_logprob(cl: Client, corpus: list[str], parallel: int = 4):
    """그 모델 토크나이저로 센 토큰 빈도(add-0.5). SLOR에서 '흔한 토큰이라 확률이 높은 것'을 빼는 데 쓴다."""
    counts: dict[int, int] = {}
    with ThreadPoolExecutor(parallel) as ex:
        for ids in ex.map(cl.tokenize, corpus):
            for t in ids:
                counts[t] = counts.get(t, 0) + 1
    n_vocab = requests.get(cl.url + "/v1/models", timeout=60).json()["data"][0]["meta"]["n_vocab"]
    total = sum(counts.values()) + 0.5 * n_vocab
    return lambda t: math.log((counts.get(t, 0) + 0.5) / total)


def score_pairs(cl: Client, pairs: list[dict], corpus: list[str], parallel: int = 4) -> list[dict]:
    """세 가지 점수로 자연스러운 쪽을 고르는지 본다.
    - sum: 문장 전체 로그확률 (짧은 문장에 유리. 번역투는 대개 길어서 자연문 쪽으로 기움)
    - char: 글자당 로그확률 (뻔한 조사·어미가 늘어나는 번역투에 유리)
    - slor: (로그확률 - 토큰 빈도만으로 본 로그확률) / 토큰 수. 길이와 흔한 토큰 효과를 함께 뺀 문법성 점수
      (Lau et al. 2017, SLOR). 주 지표.
    """
    ctx = cl.tokenize("", add_bos=True) or cl.tokenize("\n")
    uni = unigram_logprob(cl, corpus, parallel)

    def one(p):
        row = {"cat": p["cat"]}
        for k in ("natural", "translationese"):
            ids = cl.tokenize(p[k])
            lp = sentence_logprob(cl, ctx, ids)
            chars = len(p[k].replace(" ", ""))
            row[k] = {"logprob": lp, "tokens": len(ids), "chars": chars, "bpc": -lp / chars / math.log(2),
                      "slor": (lp - sum(uni(t) for t in ids)) / len(ids)}
        n, t = row["natural"], row["translationese"]
        row["pref_sum"] = n["logprob"] > t["logprob"]
        row["pref_char"] = n["bpc"] < t["bpc"]
        row["pref_slor"] = n["slor"] > t["slor"]
        return row

    with ThreadPoolExecutor(parallel) as ex:
        return list(ex.map(one, pairs))


def summarize_pairs(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        return {"n": n, "acc_slor": sum(r["pref_slor"] for r in rs) / n, "acc_sum": sum(r["pref_sum"] for r in rs) / n,
                "acc_char": sum(r["pref_char"] for r in rs) / n,
                "margin_slor": sum(r["natural"]["slor"] - r["translationese"]["slor"] for r in rs) / n}

    out = agg(rows)
    out["by_cat"] = {c: agg([r for r in rows if r["cat"] == c]) for c in sorted({r["cat"] for r in rows})}
    return out


# ---------------------------------------------------------------- 생성 통계

# 번역투 표지 (생성한 글에서 한글 1000자당 몇 번 나오나). data/translationese_pairs.tsv 의 유형과 맞춤
MARKERS = {
    "pronoun": r"그녀|그들[은이을의에]|(?<![가-힣])그(는|의|가|를|에게)(?![가-힣])|그것[은을이]",
    "have": r"[을를] (가지고|갖고) 있|[을를] (가집|가졌|가진|가지며|가지는|가지자|갖는)",
    "by_passive": r"에 의해|에 의하여",
    "double_passive": r"되어[지진질졌집]|(열려|잊혀|믿겨|쓰여|보여|짜여|나뉘어|놓여|닫혀|읽혀|불려)[지진질졌집]|불리[우운웠]",
    "about": r"에 대해|에 대한|에 대하여|께 대해",
    "through": r"[을를] 통해|[을를] 통하여",
    "in_regard": r"에 있어서|에게 있어서",
    "one_of": r"중 ?의? (하나|한 명|하루)",
    "progressive": r"고 있는 중|는 중이|는 중에 있",
    "need": r"필요로 [하한해했]",
    "causative": r"게 만[들든드]|로 하여금",
    "from": r"(으로|로|에게|에게서)부터",
    "despite": r"에도 불구하고",
}
POLITE = ("니다", "니까", "세요", "셔요", "에요", "예요", "어요", "아요", "해요", "죠", "요", "시오", "십시오")
CASUAL = ("다", "어", "아", "야", "지", "네", "래", "게", "자", "냐", "니", "까", "군", "구나", "거든", "잖아", "해", "돼",
          "줘", "봐", "라", "나", "걸", "던데", "는데", "을걸", "ㅋ", "ㅎ")
HANGUL = re.compile(r"[가-힣]")
HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def sentence_endings(text: str) -> list[str]:
    """문장 끝 어절의 끝 글자들 (문장부호·이모지 뗌)."""
    out = []
    for s in re.split(r"(?<=[.!?。])\s+|\n+", text):
        w = re.sub(r"[^가-힣ㅋㅎ]+$", "", s.strip())
        if re.search(r"[가-힣]$|[ㅋㅎ]$", w):
            out.append(w.split()[-1] if w.split() else w)
    return out


def register_of(word: str) -> str:
    if word.endswith(POLITE):
        return "polite"
    if word.endswith(CASUAL):
        return "casual"
    return "other"


def text_stats(text: str) -> dict:
    hangul = len(HANGUL.findall(text))
    letters = hangul + len(HAN.findall(text)) + len(LATIN.findall(text))
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    regs = [register_of(w) for w in sentence_endings(text)]
    return {"chars": len(text), "hangul": hangul,
            "hangul_ratio": hangul / letters if letters else 0.0,
            "han_ratio": len(HAN.findall(text)) / letters if letters else 0.0,
            "latin_ratio": len(LATIN.findall(text)) / letters if letters else 0.0,
            "rep4": 1 - len(set(grams)) / len(grams) if grams else 0.0,
            "markers": {k: len(re.findall(p, text)) for k, p in MARKERS.items()},
            "endings": {r: regs.count(r) for r in ("polite", "casual", "other")}}


def generate(cl: Client, prompts: list[dict], n_predict: int = 512, parallel: int = 4, seed: int = 0) -> list[dict]:
    def one(p):
        r = cl.post("/completion", {"prompt": cl.tokenize(cl.chat_prompt(p["prompt"]), special=True),
                                    "n_predict": n_predict, "temperature": 0.7, "top_p": 0.9, "seed": seed,
                                    "cache_prompt": False})
        return {**p, "output": r["content"], "eos": r.get("stop_type") == "eos", "stats": text_stats(r["content"])}

    with ThreadPoolExecutor(parallel) as ex:
        return list(ex.map(one, prompts))


def marker_rates(stats: list[dict]) -> dict:
    """한글 1000자당 표지 수 (전체 글을 합쳐서)."""
    hangul = sum(s["hangul"] for s in stats) or 1
    rates = {k: 1000 * sum(s["markers"][k] for s in stats) / hangul for k in MARKERS}
    rates["total"] = sum(rates.values())
    return rates


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 95% 신뢰구간."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def report(results: list[dict]) -> str:
    """results/lingbench/*.json 들을 한 표로 (마크다운). 정답률·선호율 옆 괄호는 95% 신뢰구간."""
    def pct(acc, n):
        lo, hi = wilson(acc * n, n)
        return f"{acc*100:.1f} ({lo*100:.0f}–{hi*100:.0f})"

    head = ["모델", "KoBALT", "KoGEM", "CLIcK 언어", "번역투 최소쌍 SLOR", "번역투 표지/1천자", "반말 지시", "존댓말 지시",
            "끝맺음", "4어절 반복", "한자 섞임"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        m, p, g = r.get("mcq", {}), r.get("pairs"), r.get("gen")
        cell = [r["name"]]
        cell += [pct(m[t]["acc"], m[t]["n"]) if t in m else "" for t in ("kobalt", "kogem", "click_lang")]
        cell.append(pct(p["acc_slor"], p["n"]) if p else "")
        if g:
            cell += [f"{g['markers_per_1k']['total']:.2f}",
                     f"{g.get('register_casual', {}).get('compliance', 0)*100:.0f}%",
                     f"{g.get('register_polite', {}).get('compliance', 0)*100:.0f}%",
                     f"{g['eos_rate']*100:.0f}%", f"{g['rep4']*100:.1f}%", f"{g['han_ratio']*100:.2f}%"]
        else:
            cell += [""] * 6
        rows.append("| " + " | ".join(cell) + " |")
    chance = next((r["mcq"] for r in results if r.get("mcq")), None)
    note = ""
    if chance:
        note = "\n\n찍기 기준: " + ", ".join(f"{t} {s['chance']*100:.1f}%" for t, s in chance.items())
    return "\n".join(rows) + note


def summarize_gen(rows: list[dict]) -> dict:
    st = [r["stats"] for r in rows]
    n = len(rows)
    out = {"n": n, "eos_rate": sum(r["eos"] for r in rows) / n,
           "mean_chars": sum(s["chars"] for s in st) / n,
           "hangul_ratio": sum(s["hangul_ratio"] for s in st) / n,
           "han_ratio": sum(s["han_ratio"] for s in st) / n,
           "latin_ratio": sum(s["latin_ratio"] for s in st) / n,
           "rep4": sum(s["rep4"] for s in st) / n,
           "markers_per_1k": marker_rates(st)}
    for reg in ("casual", "polite"):
        rs = [r["stats"]["endings"] for r in rows if r.get("register") == reg]
        tot = sum(e["polite"] + e["casual"] for e in rs)
        if rs:
            out[f"register_{reg}"] = {"n": len(rs), "compliance": sum(e[reg] for e in rs) / tot if tot else 0.0}
    return out
