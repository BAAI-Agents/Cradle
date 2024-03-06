from typing import (
    List,
    Dict,
    Union,
    Optional,
)
from dataclasses import dataclass, fields

import time
import json
import os

from cradle.config import Config
from cradle.log import Logger
from cradle.provider.base_embedding import EmbeddingProvider
from cradle.memory.base import BaseMemory, Image
from cradle.memory import VectorStore

cfg = Config()
logger = Logger()


@dataclass
class ConversationUnit:
    """A basic unit of memory.

    Attributes:
        messages: The messages of the conversation input.
        response: The response of the language model.
    Example Usage:
        mu = MemoryUnit(
            messages=[
                {
                    "role": "user",
                    "text": "Hello, I am a user.",
                },
            ],
            response={
                "role": "system",
                "text": "Hello, I am a system.",
            },
        )
    """
    messages: str
    response: str

    def __iter__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            yield field.name, value


class ConversationMemory(BaseMemory):
    def __init__(
        self,
        memory_path: str,
        vectorstores: Dict[str, VectorStore],
        embedding_provider: EmbeddingProvider,
        memory: Optional[Dict] = None,
    ) -> None:
        if memory is None:
            self.memory: Dict[str, ConversationUnit] = {}
        else:
            self.memory = memory
        self.memory_path = memory_path
        self.vectorstores = vectorstores
        self.embedding_provider = embedding_provider

    def add(
        self,
        messages: str,
        response: str,
        **kwargs,
    ) -> None:
        """Add data to memory.

        Args:
            messages: the messages of the conversation input.
            response: the response of the language model.
            **kwargs: Other keyword arguments that subclasses might use.
        """
        name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())  # the unique id of the added unit.
        mem_unit = ConversationUnit(
            messages=messages,
            response=response,
        )
        self.memory[name] = mem_unit
        embeddings = self.embedding_provider.embed_query(mem_unit.messages)
        self.vectorstores["message"].add_embeddings([name], [embeddings])

    def similarity_search(
        self,
        query: Union[str, Image],
        top_k: int = 3,
        **kwargs,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the vectorstores.

        Args:
            query: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """
        query_embedding = self.embedding_provider.embed_query(query)
        key_and_score = self.vectorstores["message"].similarity_search(query_embedding, top_k)

        return [self.memory[k] for k, score in key_and_score]


    def load(
        cls,
        memory_path: str,
        vectorstores: Dict[str, VectorStore],
        embedding_provider: EmbeddingProvider,
    ) -> "ConversationMemory":
        """Load the memory from the local file."""
        with open(os.path.join(memory_path, "memory.json"), "r") as rf:
            memory = json.load(rf)

        return cls(
            memory_path=memory_path,
            vectorstores=vectorstores,
            embedding_provider=embedding_provider,
            memory=memory,
        )


    def save(self) -> None:
        """Save the memory to the local file."""
        with open(os.path.join(self.memory_path, "memory.json"), "w") as f:
            json.dump(self.memory, f, indent=2)
        for k, v in self.vectorstores.items():
            v.save(name=k)
