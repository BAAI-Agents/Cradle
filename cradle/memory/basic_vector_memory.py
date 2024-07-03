from typing import (
    List,
    Dict,
    Union,
    Optional,
    Any
)
import os

from cradle.config import Config
from cradle.log import Logger
from cradle.memory.base import BaseMemory, Image
from cradle.memory.vector_store import VectorStore
from cradle.utils.json_utils import load_json, save_json

config = Config()
logger = Logger()


class BasicVectorMemory(BaseMemory):

    storage_filename = "vector_memory.json"

    def __init__(
        self,
        memory_path: str,
        vectorstores: VectorStore,
        embedding_provider: Any,
        memory: Optional[Dict] = None,
    ):
        if memory is None:
            self.memory: Dict = {}
        else:
            self.memory = memory
        self.memory_path = memory_path
        self.vectorstores = vectorstores
        self.embedding_provider = embedding_provider


    def add(
        self,
        data: Dict[str, Union[str, Image]],
    ) -> None:
        """Add data to memory.

        Args:
            data: the mapping from unique name (id) to text/image.
        """

        keys: List[str] = list(data.keys())
        embeddings = []

        for k in keys:
            embeddings.append(self.embedding_provider.embed_query(data[k]["description"]))
            instruction = data[k]["instruction"]
            screenshot = data[k]["screenshot"]
            timestep = data[k]["timestep"]
            description = data[k]["description"]
            inventory = data[k]["inventory"]

            self.memory[k] = {
                "instruction": instruction,
                "screenshot": screenshot,
                "timestep": timestep,
                "description": description,
                "inventory": inventory,
            }

        self.vectorstores['description'].add_embeddings(keys, embeddings)


    def similarity_search(
        self,
        data: Union[str, Image],
        top_k: int = 3,
        **kwargs,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the vectorstore.

        Args:
            data: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """
        query_embedding = self.embedding_provider.embed_query(data)
        key_and_score = self.vectorstores['description'].similarity_search(query_embedding, top_k)

        return [self.memory[k] for k, score in key_and_score]


    def recent_search(
        self,
        recent_k: int = 3,
        **kwargs,
    ) -> List[Union[str, Image]]:
        """Retrieve the recent k keys

        Args:
            recent_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values of the recent k memory.
        """

        keys = list(self.memory.keys()) # the order of adding
        recent_k = min(recent_k,len(keys))
        return [self.memory[k] for k in keys[len(keys) - recent_k : len(keys)]]


    def load(self):
        self.load()


    def load(
        cls,
        memory_path: str,
        vectorstore: VectorStore,
        embedding_provider: Any,
    ) -> "BasicVectorMemory":
        """Load the memory from the local file."""

        memory = load_json(os.path.join(cls.memory_path, cls.storage_filename))

        return cls(
            memory_path=memory_path,
            vectorstore=vectorstore,
            embedding_provider=embedding_provider,
            memory=memory,
        )


    def save(self) -> None:
        """Save the memory to the local file."""
        save_json(file_path = os.path.join(self.memory_path, self.storage_filename), json_dict = self.memory, indent = 4)
        self.vectorstores.save()
