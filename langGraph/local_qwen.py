"""Thread-safe, fully offline Qwen2.5 text-generation runtime."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_paths import QWEN_MODEL_DIR, resolve_qwen_model_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN_ROOT = QWEN_MODEL_DIR


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class LocalQwenRuntime:
    """Lazy resident Qwen runtime used only for final grounded generation."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_QWEN_ROOT,
        *,
        device: str = "auto",
        max_new_tokens: int = 768,
        eager_load: bool = False,
    ) -> None:
        if max_new_tokens < 32:
            raise ValueError("max_new_tokens must be >= 32")
        self.model_path = resolve_qwen_model_path(model_path)
        self.device = resolve_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.RLock()
        if eager_load:
            self.load()

    def load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            if self.device.startswith("cuda"):
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                dtype = torch.float32
            load_kwargs: dict[str, Any] = {"local_files_only": True}
            dtype_key = (
                "dtype"
                if int(transformers.__version__.split(".", 1)[0]) >= 5
                else "torch_dtype"
            )
            load_kwargs[dtype_key] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **load_kwargs
            ).to(self.device)
            self.model.eval()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )[0].strip()
