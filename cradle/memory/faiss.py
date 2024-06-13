from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import os
import numpy as np
import pickle
from pathlib import Path

from cradle.memory import VectorStore
from cradle.provider.base_embedding import EmbeddingProvider
from cradle.config import Config
from cradle.log import Logger

config = Config()
logger = Logger()


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.
    Code borrowed from langchain.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class FAISS(VectorStore):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        memory_path: str,
        index: Optional[Any] = None,
        index_to_key: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize the Meta Faiss vectorstore.
        Code modified based on langchain.

        Args:
            embedding_provider: Embedding provider.
            memory_path: Path to the store memory.
            index: Faiss index.
            index_to_key: Mapping from index to key.
        """
        faiss = dependable_faiss_import()
        if index is None:
            self.index = faiss.IndexFlatL2(embedding_provider.get_embedding_dim())
        if index_to_key is None:
            self.index_to_key = {}
        self.memory_path = memory_path

    def add_embeddings(
        self,
        keys: List[str],
        embeddings: List[List[float]],
        **kwargs,
    ) -> None:
        """Add embeddings to the vectorstore.

        Args:
            keys: List of metadatas associated with the embedding.
            embeddings: List of embeddings to add to the vectorstore.
            **kwargs: Other keyword arguments.
        """
        assert len(keys) == len(
            embeddings
        ), f"keys: {len(keys)}, embeddings: {len(embeddings)} expected to be equal length"

        vector = np.array(embeddings, dtype=np.float32)
        self.index.add(vector)

        starting_len = len(self.index_to_key)
        index_to_key = {starting_len + j: id_ for j, id_ in enumerate(keys)}
        self.index_to_key.update(index_to_key)

    def delete(
        self,
        keys: List[str] = None,
        **kwargs,
    ) -> bool:
        """Delete by keys.

        Args:
            keys: List of keys to delete.
            **kwargs: Other keyword arguments.

        Returns:
            bool: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        missing_keys = set(keys).difference(self.index_to_key.values())
        if missing_keys:
            raise ValueError(
                f"Some specified keys do not exist in the current store: "
                f"{missing_keys}"
            )

        reversed_index = {id_: idx for idx, id_ in self.index_to_key.items()}
        index_to_delete = [reversed_index[id_] for id_ in keys]
        self.index.remove_ids(np.array(index_to_delete, dtype=np.int64))

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_key.items())
            if i not in index_to_delete
        ]
        self.index_to_key = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True

    def update(
        self,
        keys: List[str],
        embeddings: List[List[float]],
        **kwargs,
    ) -> None:
        """Update embeddings to the vectorstore.

        Args:
            keys: List of metadatas associated with the embedding.
            embeddings: List of embeddings to add to the vectorstore.
            **kwargs: Other keyword arguments.
        """
        self.delete(keys)
        self.add_embeddings(keys, embeddings)

    def similarity_search(
        self,
        embedding: List[float],
        top_k: int,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """Return keys most similar to query.

        Args:
            embedding: Query embedding.
            top_k: Number of keys to return.
            **kwargs: Other keyword arguments.

        Returns:
            List of (key, score) tuples.
        """
        vector = np.array([embedding], dtype=np.float32)
        scores, indices = self.index.search(vector, min(top_k, len(self.index_to_key)))

        key_and_score = []
        for idx, score in zip(indices[0], scores[0]):
            key_and_score.append((self.index_to_key[idx], score))

        return key_and_score


    def load(
        cls,
        embedding_provider: EmbeddingProvider,
        memory_path: str,
        name: str,
    ) -> "FAISS":
        """Load FAISS index and index_to_key from disk.

        Args:
            embedding_provider: Embeddings to use when generating queries
            memory_path: folder path to load index and index_to_key from.
            name: name of the vectorstore.

        Returns:
            The FAISS vectorstore class.
        """
        path = Path(memory_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / f"{name}.faiss"))
        # load index_to_key
        with open(path / f"{name}.pkl", "rb") as f:
            index_to_key = pickle.load(f)

        return cls(
            embedding_provider=embedding_provider,
            memory_path=memory_path,
            index=index,
            index_to_key=index_to_key,
        )


    def save(self, name: str) -> None:
        """Save FAISS index and index_to_key to disk."""
        path = Path(self.memory_path)
        path.mkdir(exist_ok=True, parents=True)
        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / f"{name}.faiss"))
        # save index_to_key
        with open(path / f"{name}.pkl", "wb") as f:
            pickle.dump(self.index_to_key, f)
