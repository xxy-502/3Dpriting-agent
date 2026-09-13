"""Offline Dense + BM25 + RRF retrieval for the bundled Qdrant collection."""

from __future__ import annotations

import atexit
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

from .model_paths import BGE_MODEL_DIR, resolve_bge_model_path


LANGGRAPH_ROOT = Path(__file__).resolve().parent
DEFAULT_COLLECTION_DIR = LANGGRAPH_ROOT / "Qdrant_collection"
DEFAULT_BGE_MODEL_DIR = BGE_MODEL_DIR
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/+%-][a-z0-9]+)*")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_for_tokens(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    replacements = {
        "°c": " degc ",
        "°f": " degf ",
        "µm": " um ",
        "μm": " um ",
        "mm³": " mm3 ",
        "cm³": " cm3 ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def tokenize(text: str) -> list[str]:
    """Match the tokenizer used when the persisted BM25 vectors were built."""

    return TOKEN_RE.findall(_normalize_for_tokens(text))


class BGEEncoder:
    """Minimal local BGE encoder using CLS pooling and L2 normalization."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Embedding model directory not found: {model_path}")
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
            if device == "auto"
            else device
        )
        self.device = torch.device(resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        self.max_length = min(
            int(getattr(self.model.config, "max_position_embeddings", 512)), 512
        )

    def encode(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = values[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: tensor.to(self.device) for key, tensor in encoded.items()
                }
                embeddings = self.model(**encoded).last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                vectors.append(
                    embeddings.cpu().numpy().astype(np.float32, copy=False)
                )
        return np.concatenate(vectors, axis=0)


def _sparse_query(
    text: str, vocabulary: dict[str, Any]
) -> models.SparseVector:
    token_to_index = vocabulary["token_to_index"]
    indices = sorted(
        {
            token_to_index[token]
            for token in tokenize(text)
            if token in token_to_index
        }
    )
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))


class OfflineHybridRetriever:
    """One-process owner of the copied persistent Qdrant collection."""

    def __init__(
        self,
        collection_dir: Path | str = DEFAULT_COLLECTION_DIR,
        *,
        bge_model_path: Path | str = DEFAULT_BGE_MODEL_DIR,
        device: str = "cpu",
    ) -> None:
        self.collection_dir = Path(collection_dir).expanduser().resolve()
        config_path = self.collection_dir / "collection_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qdrant configuration not found: {config_path}")
        self.config = _load_json(config_path)

        dense = self.config["vectors"]["dense"]
        sparse = self.config["vectors"]["sparse"]
        self.bge_model_path = resolve_bge_model_path(bge_model_path)
        self.encoder = BGEEncoder(self.bge_model_path, device=device)
        if self.encoder.dimension != int(dense["dimension"]):
            raise ValueError(
                "Embedding dimension does not match collection configuration: "
                f"{self.encoder.dimension} != {dense['dimension']}"
            )
        self.query_instruction = str(dense["query_instruction"])
        self.vocabulary = _load_json(
            self.collection_dir / sparse["vocabulary_path"]
        )
        storage = self.collection_dir / self.config["storage_path"]
        self.client = QdrantClient(path=str(storage))
        self._lock = threading.RLock()
        self._closed = False
        atexit.register(self.close)

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        prefetch_limit: int | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("RAG query must be a non-empty string")
        result_limit = int(
            limit or self.config["fusion"]["default_result_limit"]
        )
        prefetch = int(
            prefetch_limit or self.config["fusion"]["prefetch_limit"]
        )
        if result_limit < 1 or prefetch < result_limit:
            raise ValueError("prefetch_limit must be >= limit >= 1")

        dense_vector = self.encoder.encode(
            [self.query_instruction + query], batch_size=1
        )[0].tolist()
        sparse_vector = _sparse_query(query, self.vocabulary)
        requests: list[models.Prefetch] = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch,
                filter=query_filter,
            )
        ]
        if sparse_vector.indices:
            requests.append(
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch,
                    filter=query_filter,
                )
            )

        with self._lock:
            response = self.client.query_points(
                collection_name=self.config["collection_name"],
                prefetch=requests,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=result_limit,
                with_payload=True,
                with_vectors=False,
            )

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_type": payload.get("chunk_type"),
                    "page_content": payload.get("page_content", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return results

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.client.close()
            self._closed = True


class LangChainPrintingRetriever(BaseRetriever):
    """LangChain Retriever facade for optional use outside the main graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: OfflineHybridRetriever
    result_limit: int = Field(default=5, ge=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        del run_manager
        rows = self.backend.search(query, limit=self.result_limit)
        return [
            Document(
                page_content=row["page_content"],
                metadata={
                    **row["metadata"],
                    "score": row["score"],
                    "chunk_id": row["chunk_id"],
                    "chunk_type": row["chunk_type"],
                },
            )
            for row in rows
        ]
