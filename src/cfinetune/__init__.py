"""gpt-oss MoE 한국어 경량화 파이프라인: 전문가 프루닝 -> 양자화 -> 저랭크 추가학습 -> 평가/경제성."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
