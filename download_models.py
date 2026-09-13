"""Download the official model weights required by the offline Agent.

The custom YOLO checkpoint is intentionally not downloaded or overwritten.
Run this script from any working directory; paths are resolved relative to the
directory containing this file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"

QWEN_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_TARGET = MODEL_ROOT / "Qwen" / "Qwen2.5-3B-Instruct"
QWEN_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "README.md",
    "LICENSE*",
)

BGE_REPO_ID = "BAAI/bge-small-en-v1.5"
BGE_TARGET = MODEL_ROOT / "bge" / "bge-small-en-v1.5"
BGE_ALLOW_PATTERNS = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
    "README.md",
    "LICENSE*",
)

SAM_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_base_plus.pt"
)
SAM_TARGET = MODEL_ROOT / "sam" / "sam2.1_hiera_base_plus.pt"

ALL_MODELS = ("qwen", "bge", "sam")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=ALL_MODELS,
        help="Models to download; omit to download qwen, bge, and sam.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the official sources and target paths without downloading.",
    )
    return parser


def _print_plan(selected: Iterable[str]) -> None:
    print("Model download plan:")
    for name in selected:
        if name == "qwen":
            print(f"  Qwen: hf://{QWEN_REPO_ID} -> {QWEN_TARGET}")
        elif name == "bge":
            print(f"  BGE:  hf://{BGE_REPO_ID} -> {BGE_TARGET}")
        elif name == "sam":
            print(f"  SAM:  {SAM_URL} -> {SAM_TARGET}")
    print(f"  YOLO: preserved at {MODEL_ROOT / 'yolo' / 'yolo_best.pt'}")


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required. Install it with: "
            "python -m pip install huggingface-hub==1.15.0"
        ) from error
    return snapshot_download


def _download_hugging_face_model(
    *,
    label: str,
    repo_id: str,
    target: Path,
    allow_patterns: tuple[str, ...],
    force: bool,
) -> None:
    snapshot_download = _load_snapshot_download()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} from official repository {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        revision="main",
        local_dir=str(target),
        allow_patterns=list(allow_patterns),
        force_download=force,
        token=os.environ.get("HF_TOKEN") or None,
    )
    _validate_transformer_model(target, label)
    print(f"{label} ready: {target}")


def _validate_transformer_model(path: Path, label: str) -> None:
    has_config = (path / "config.json").is_file()
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    )
    has_weights = any(path.glob("*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )
    if not (has_config and has_tokenizer and has_weights):
        raise RuntimeError(
            f"{label} download is incomplete at {path}; expected config, "
            "tokenizer, and model weight files"
        )


def _download_file(url: str, target: Path, *, force: bool) -> None:
    if target.is_file() and target.stat().st_size > 0 and not force:
        print(f"SAM checkpoint already exists, skipping: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "3Dprint-detectagent-model-downloader/1.0"},
    )

    for attempt in range(1, 4):
        try:
            if partial.exists():
                partial.unlink()
            print(f"Downloading SAM 2.1 Hiera Base+ (attempt {attempt}/3) ...")
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "wb"
            ) as output:
                total = int(response.headers.get("Content-Length", "0"))
                received = 0
                last_percent = -1
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if total:
                        percent = int(received * 100 / total)
                        if percent >= last_percent + 5 or percent == 100:
                            print(f"  {percent:3d}% ({received / 1024**2:.1f} MiB)")
                            last_percent = percent
            if partial.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            partial.replace(target)
            print(f"SAM ready: {target}")
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main() -> int:
    args = _build_parser().parse_args()
    selected = tuple(dict.fromkeys(args.models or ALL_MODELS))
    _print_plan(selected)
    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return 0

    try:
        if "qwen" in selected:
            _download_hugging_face_model(
                label="Qwen2.5-3B-Instruct",
                repo_id=QWEN_REPO_ID,
                target=QWEN_TARGET,
                allow_patterns=QWEN_ALLOW_PATTERNS,
                force=args.force,
            )
        if "bge" in selected:
            _download_hugging_face_model(
                label="BGE Small EN v1.5",
                repo_id=BGE_REPO_ID,
                target=BGE_TARGET,
                allow_patterns=BGE_ALLOW_PATTERNS,
                force=args.force,
            )
        if "sam" in selected:
            _download_file(SAM_URL, SAM_TARGET, force=args.force)
    except KeyboardInterrupt:
        print("Download cancelled by user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    print("All requested models are ready.")
    print("Run the Agent with: python -m langGraph.agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
