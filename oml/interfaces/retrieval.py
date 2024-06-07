from abc import abstractmethod
from typing import Any


class IRetrievalPostprocessor:
    """
    This is a base interface for the classes which somehow postprocess retrieval results.

    """

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:  # type: ignore
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
