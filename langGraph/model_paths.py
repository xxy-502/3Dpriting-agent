"""Deterministic discovery of local model files used by the agent."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models"
QWEN_MODEL_DIR = MODEL_ROOT / "Qwen"
YOLO_MODEL_DIR = MODEL_ROOT / "yolo"
SAM_MODEL_DIR = MODEL_ROOT / "sam"
BGE_MODEL_DIR = MODEL_ROOT / "bge"


def resolve_pt_checkpoint(
    checkpoint: Path | str | None,
    *,
    model_dir: Path,
    model_type: str,
    preferred_names: tuple[str, ...] = (),
) -> Path:
    """Resolve a checkpoint file or deterministically select a ``.pt`` file."""

    candidate = Path(checkpoint).expanduser() if checkpoint is not None else model_dir
    candidate = candidate.resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".pt":
        return candidate

    search_dir = candidate if candidate.is_dir() else None
    matches = (
        sorted(
            (
                path.resolve()
                for path in search_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".pt"
            ),
            key=lambda path: path.as_posix().lower(),
        )
        if search_dir is not None
        else []
    )
    if not matches:
        raise FileNotFoundError(
            f"No {model_type} .pt checkpoint found in: {search_dir or candidate}"
        )

    preferred_order = {name.lower(): index for index, name in enumerate(preferred_names)}
    return min(
        matches,
        key=lambda path: (
            preferred_order.get(path.name.lower(), len(preferred_order)),
            path.as_posix().lower(),
        ),
    )


def resolve_yolo_checkpoint(checkpoint: Path | str | None = None) -> Path:
    return resolve_pt_checkpoint(
        checkpoint,
        model_dir=YOLO_MODEL_DIR,
        model_type="YOLO",
        preferred_names=("yolo_best.pt", "best.pt"),
    )


def resolve_sam_checkpoint(checkpoint: Path | str | None = None) -> Path:
    return resolve_pt_checkpoint(
        checkpoint,
        model_dir=SAM_MODEL_DIR,
        model_type="SAM",
        preferred_names=("sam_best.pt", "best.pt"),
    )


def _is_transformer_model_dir(path: Path) -> bool:
    has_config = (path / "config.json").is_file()
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer_config.json", "tokenizer.json")
    )
    has_weights = any(path.glob("*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )
    return has_config and has_tokenizer and has_weights


def resolve_qwen_model_path(model_path: Path | str = QWEN_MODEL_DIR) -> Path:
    """Find a complete local Qwen model rooted at ``models/Qwen`` by default."""

    root = Path(model_path).expanduser().resolve()
    candidates = [root, root / "snapshots" / "master"]
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted(path for path in snapshots.iterdir() if path.is_dir()))
    if root.is_dir():
        candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_transformer_model_dir(resolved):
            return resolved

    raise FileNotFoundError(
        "No complete local Qwen model found under "
        f"{root}; expected config.json, tokenizer files, and model weights"
    )


def resolve_bge_model_path(model_path: Path | str = BGE_MODEL_DIR) -> Path:
    """Find a complete local BGE model rooted at ``models/bge`` by default."""

    root = Path(model_path).expanduser().resolve()
    candidates = [root, root / "snapshots" / "master"]
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted(path for path in snapshots.iterdir() if path.is_dir()))
    if root.is_dir():
        candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_transformer_model_dir(resolved):
            return resolved

    raise FileNotFoundError(
        "No complete local BGE model found under "
        f"{root}; expected config.json, tokenizer files, and model weights"
    )
