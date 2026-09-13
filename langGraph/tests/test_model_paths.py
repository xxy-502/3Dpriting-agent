from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langGraph.model_paths import (
    resolve_bge_model_path,
    resolve_pt_checkpoint,
    resolve_qwen_model_path,
)


class ModelPathTests(unittest.TestCase):
    def test_pt_discovery_prefers_deployment_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "other.pt").touch()
            preferred = root / "best.pt"
            preferred.touch()

            result = resolve_pt_checkpoint(
                None,
                model_dir=root,
                model_type="test",
                preferred_names=("best.pt",),
            )

            self.assertEqual(result, preferred.resolve())

    def test_pt_discovery_accepts_any_pt_and_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            nested = root / "version-1" / "custom.pt"
            nested.parent.mkdir()
            nested.touch()

            result = resolve_pt_checkpoint(
                None, model_dir=root, model_type="test"
            )

            self.assertEqual(result, nested.resolve())

    def test_pt_discovery_fails_only_when_no_pt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "notes.txt").touch()

            with self.assertRaisesRegex(FileNotFoundError, r"No test \.pt"):
                resolve_pt_checkpoint(None, model_dir=root, model_type="test")

    def test_qwen_model_can_be_directly_under_qwen_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (root / name).touch()

            self.assertEqual(resolve_qwen_model_path(root), root.resolve())

    def test_qwen_model_can_be_in_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            snapshot = root / "snapshots" / "master"
            snapshot.mkdir(parents=True)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "model-00001-of-00002.safetensors",
            ):
                (snapshot / name).touch()

            self.assertEqual(resolve_qwen_model_path(root), snapshot.resolve())

    def test_bge_model_can_be_nested_under_bge_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model = root / "bge-small-en-v1.5"
            model.mkdir()
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (model / name).touch()

            self.assertEqual(resolve_bge_model_path(root), model.resolve())


if __name__ == "__main__":
    unittest.main()
