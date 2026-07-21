"""
BGE embedding wrapper around sentence-transformers.

Uses BAAI/bge-small-en-v1.5 (384-dim, cached in ~/.cache/huggingface/).
The model is loaded once and reused across all calls.

BGE is asymmetric: queries get an instruction prefix, passages do not.
`embed_query` and `embed_passages` enforce that split so callers cannot
accidentally mix them up.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from research_assistant.config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


class Embedder:
    def __init__(self) -> None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.")

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return int(self._model.get_embedding_dimension())
        return int(self._model.get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, with the BGE retrieval prefix applied."""
        vector = self._model.encode(
            BGE_QUERY_PREFIX + text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages. No prefix - BGE expects passages raw."""
        vectors = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]

    def embed_passage(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]
