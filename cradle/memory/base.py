import abc
from typing import (
    Any,
    Iterable,
    List,
    Dict,
    Union,
    Tuple,
    Optional,
)

from cradle.config.config import Config

Image = Any

config = Config()


class BaseMemory:
    """Base class for all memories."""

    @abc.abstractmethod
    def add(
        self,
        **kwargs,
    ) -> None:
        """Add data to memory.

        Args:
            **kwargs: Other keyword arguments that subclasses might use.
        """
        pass


    @abc.abstractmethod
    def similarity_search(
        self,
        data: Union[str, Image],
        top_k: int,
        **kwargs: Any,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the store.

        Args:
            data: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """
        pass


    @abc.abstractmethod
    def add_recent_history(
        self,
        key: str,
        info: Any,
    ) -> None:
        pass


    @abc.abstractmethod
    def get_recent_history(
        self,
        key: str,
        k: int = 1,
    ) -> List[Any]:
        pass


    @abc.abstractmethod
    def add_summarization(self, hidden_state: str) -> None:
        pass


    @abc.abstractmethod
    def get_summarization(self) -> str:
        pass


    @abc.abstractmethod
    def load(self) -> None:
        """Load the memory from persistence."""


    @abc.abstractmethod
    def save(self) -> None:
        """Save the memory to persistence."""
