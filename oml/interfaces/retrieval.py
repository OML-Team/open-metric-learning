from abc import abstractmethod
from typing import Any


class IRetrievalPostprocessor:
    """
    This is a base interface for the classes which somehow postprocess retrieval results.

    """

    @property
    def top_n(self) -> int:
        """
        Returns: Number of first n items to process.

        """
        raise NotImplementedError()

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:  # type: ignore
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
