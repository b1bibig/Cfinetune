import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfinetune import load_config  # noqa: E402

RESULTS = ROOT / "results"


def resolve_model_dir(model: str, local_root: str | Path = "/content/models") -> Path:
    """로컬 폴더면 그대로, 아니면 허깅페이스에서 받는다 (원본 MXFP4 HF 형식만, original/ metal/ 제외)."""
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    dst = Path(local_root) / model.replace("/", "__")
    snapshot_download(model, local_dir=dst, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
                      ignore_patterns=["original/*", "metal/*"])
    return dst


__all__ = ["ROOT", "RESULTS", "load_config", "resolve_model_dir"]
