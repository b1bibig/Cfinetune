"""llama.cpp 실행 도구: 서버로 객관식 채점, 퍼플렉시티, 속도/전력 측정.

모든 변형(원본, 프루닝, 양자화, LoRA)을 같은 GGUF + 같은 엔진으로 재야 공정한 비교가 된다.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .prompts import chat_prompt, mcq_prompt


def _bin(bin_dir: str | Path, name: str) -> str:
    """llama.cpp 실행 파일 경로 (윈도우는 .exe)."""
    return str(Path(bin_dir) / (name + (".exe" if os.name == "nt" else "")))


class LlamaServer:
    def __init__(self, bin_dir: str | Path, model: str | Path, port: int = 8090, ngl: int = 99,
                 parallel: int = 4, ctx_per_slot: int = 4096, log: str | Path | None = None, extra=()):
        self.cmd = [_bin(bin_dir, "llama-server"), "-m", str(model), "--port", str(port),
                    "-ngl", str(ngl), "-np", str(parallel), "-c", str(ctx_per_slot * parallel), *extra]
        self.url = f"http://127.0.0.1:{port}"
        self.parallel = parallel
        self.log = open(log or "/tmp/llama-server.log", "w")

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server 종료됨 (로그: {self.log.name})")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("llama-server 시작 시간 초과")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def complete_text(self, prompt: str, **kw) -> dict:
        """문자열 프롬프트 (특수 토큰 문자열도 llama.cpp가 특수 토큰으로 읽음)."""
        payload = {"prompt": prompt, "temperature": 0, "cache_prompt": False, **kw}
        r = requests.post(self.url + "/completion", json=payload, timeout=900)
        r.raise_for_status()
        return r.json()

    def complete(self, prompt_ids: list[int], **kw) -> dict:
        payload = {"prompt": prompt_ids, "temperature": 0, "cache_prompt": True, **kw}
        r = requests.post(self.url + "/completion", json=payload, timeout=900)
        r.raise_for_status()
        return r.json()


def _top_logprobs(resp: dict) -> dict[int, float]:
    """llama-server 버전마다 다른 응답 형식을 {token_id: logprob} 로 통일."""
    probs = resp.get("completion_probabilities") or resp.get("probs") or []
    if not probs:
        return {}
    first = probs[0]
    tops = first.get("top_logprobs") or first.get("top_probs") or first.get("probs") or []
    out = {}
    for t in tops:
        tid = t.get("id", t.get("tok_id"))
        if "logprob" in t:
            lp = t["logprob"]
        else:
            p = t.get("prob", 0.0)
            lp = math.log(p) if p > 0 else -math.inf
        if tid is not None:
            out[int(tid)] = float(lp)
    return out


def letter_token_ids(tokenizer, n: int = 5) -> list[list[int]]:
    """'A'와 ' A'처럼 알파벳 하나로 이루어진 토큰 id들."""
    out = []
    for L in "ABCDE"[:n]:
        ids = set()
        for s in (L, " " + L):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        out.append(sorted(ids))
    return out


def score_mcq(server: LlamaServer, tokenizer, arch, items: list[dict], fmt: str = "chat",
              n_probs: int = 50, progress=None) -> list[dict]:
    letter_ids = letter_token_ids(tokenizer)

    def one(item):
        ids = tokenizer(mcq_prompt(arch, item, fmt), add_special_tokens=False).input_ids
        lp = _top_logprobs(server.complete(ids, n_predict=1, n_probs=n_probs))
        scores = [max((lp.get(t, -math.inf) for t in letter_ids[i]), default=-math.inf)
                  for i in range(len(item["choices"]))]
        best = max(range(len(scores)), key=lambda i: scores[i])
        pred = best if scores[best] > -math.inf else -1
        return {"task": item["task"], "subject": item["subject"], "answer": item["answer"],
                "pred": pred, "correct": pred == item["answer"], "n_prompt": len(ids)}

    with ThreadPoolExecutor(server.parallel) as ex:
        results = []
        for i, r in enumerate(ex.map(one, items)):
            results.append(r)
            if progress and i % 200 == 0:
                progress(i, len(items))
    return results


def summarize_mcq(results: list[dict]) -> dict:
    out = {}
    for task in sorted({r["task"] for r in results}):
        rs = [r for r in results if r["task"] == task]
        out[task] = {"acc": sum(r["correct"] for r in rs) / len(rs), "n": len(rs),
                     "no_letter": sum(r["pred"] < 0 for r in rs) / len(rs)}
    return out


def generate(server: LlamaServer, tokenizer, arch, prompts: list[str], n_predict: int = 400) -> list[str]:
    outs = []
    for p in prompts:
        ids = tokenizer(chat_prompt(arch, p), add_special_tokens=False).input_ids
        outs.append(server.complete(ids, n_predict=n_predict, stop=[arch.END])["content"])
    return outs


def perplexity(bin_dir: str | Path, model: str | Path, text_file: str | Path, ctx: int = 512,
               chunks: int = 200, ngl: int = 99) -> dict:
    cmd = [_bin(bin_dir, "llama-perplexity"), "-m", str(model), "-f", str(text_file),
           "-c", str(ctx), "--chunks", str(chunks), "-ngl", str(ngl)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", p.stdout + p.stderr)
    if not m:
        raise RuntimeError("퍼플렉시티 결과를 찾지 못함:\n" + (p.stdout + p.stderr)[-2000:])
    return {"ppl": float(m.group(1)), "ppl_err": float(m.group(2)), "ctx": ctx, "chunks": chunks}


# ---------------------------------------------------------------- 속도 / 전력

class PowerSampler:
    """nvidia-smi를 0.2초마다 읽어 평균 전력(W)과 최대 VRAM(MiB)을 기록."""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, float]] = []

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}", "-i", "0"], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                w, mem = (float(x) for x in line.strip().split(","))
                self.samples.append((w, mem))
            except ValueError:
                pass

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()

    @property
    def mean_w(self) -> float:
        return sum(s[0] for s in self.samples) / max(len(self.samples), 1)

    def active_mean_w(self, idle_w: float) -> float:
        """모델 로딩처럼 GPU가 놀던 구간을 빼고, 실제 연산 중인 샘플만 평균."""
        if not self.samples:
            return 0.0
        peak = max(s[0] for s in self.samples)
        cut = idle_w + 0.3 * (peak - idle_w)
        active = [s[0] for s in self.samples if s[0] >= cut]
        return sum(active) / len(active)

    @property
    def peak_mem_mib(self) -> float:
        return max((s[1] for s in self.samples), default=0.0)


def gpu_energy_j() -> float | None:
    """GPU 자체 에너지 계량기(드라이버 시작 후 누적, Volta 이후 GPU: T4·L4·A100 등). 지원 안 하면 None."""
    p = subprocess.run(["nvidia-smi", "--query-gpu=total_energy_consumption", "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip()) / 1000  # mJ -> J
    except ValueError:
        return None


def answer_energy(server: "LlamaServer", arch, prompts: list[str], n_predict: int = 512) -> dict:
    """질문에 실제로 답하는 에너지: 질문을 하나씩(사용자 1명 기준) 보내고 GPU 에너지 계량기로 잰다.

    프롬프트는 문자열로 보내서 llama.cpp가 직접 토큰으로 자른다 (어휘가 다른 모델도 같은 방식).
    프롬프트 처리 + 답변 생성 전체를 포함한다. 계량기가 없으면 0.2초 간격 전력 기록을 적분한다.
    """
    server.complete_text(arch.chat_prefix("안녕하세요?"), n_predict=8)  # 예열 (측정 제외)
    texts, n_tok, n_prompt = [], 0, 0
    e0 = gpu_energy_j()
    with PowerSampler() as ps:
        t0 = time.time()
        for p in prompts:
            r = server.complete_text(arch.chat_prefix(p), n_predict=n_predict, stop=[arch.END])
            texts.append(r["content"])
            n_tok += r.get("tokens_predicted", 0)
            n_prompt += r.get("tokens_evaluated", 0)
        sec = time.time() - t0
    e1 = gpu_energy_j()
    joule = e1 - e0 if e0 is not None and e1 is not None else ps.mean_w * sec
    chars = sum(len(t) for t in texts)
    return {"n_answers": len(prompts), "seconds": sec, "joule": joule, "energy_source": "counter" if e0 is not None else "power",
            "mean_w": joule / sec, "gen_chars": chars, "gen_tokens": n_tok, "prompt_tokens": n_prompt,
            "wh_per_answer": joule / 3600 / len(prompts), "wh_per_1k_chars": joule / 3600 / max(chars, 1) * 1000,
            "chars_per_s": chars / sec, "answers": texts}


def gpu_name() -> str:
    p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                       capture_output=True, text=True)
    return p.stdout.strip()


def idle_power(seconds: float = 5) -> float:
    with PowerSampler() as s:
        time.sleep(seconds)
    return s.mean_w


def llama_bench(bin_dir: str | Path, model: str | Path, n_prompt: int = 512, n_gen: int = 128,
                reps: int = 10, ngl: int = 99) -> dict:
    """프롬프트 처리(pp)와 생성(tg)을 따로 재서 각각의 속도와 전력을 얻는다.

    ngl: GPU에 올릴 층 수. VRAM에 다 안 들어가는 모델은 낮춰서 나머지를 CPU로 돌린다 (전력은 GPU만 잼).
    """
    out = {"gpu": gpu_name(), "idle_w": idle_power(), "file_gb": Path(model).stat().st_size / 1e9, "ngl": ngl}
    for name, p, n in (("pp", n_prompt, 0), ("tg", 0, n_gen)):
        cmd = [_bin(bin_dir, "llama-bench"), "-m", str(model), "-ngl", str(ngl),
               "-p", str(p), "-n", str(n), "-r", str(reps), "-o", "json"]
        with PowerSampler() as s:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        rows = json.loads(proc.stdout)
        out[f"{name}_tps"] = rows[0]["avg_ts"]
        out[f"{name}_tps_std"] = rows[0]["stddev_ts"]
        out[f"{name}_power_w"] = s.active_mean_w(out["idle_w"])
        out["peak_vram_gb"] = max(out.get("peak_vram_gb", 0), s.peak_mem_mib / 1024)
        out["n_params"] = rows[0].get("model_n_params")
        out["model_size_gb"] = rows[0].get("model_size", 0) / 1e9
    return out
