from typing import Any

from oml.interfaces.retrieval import IRetrievalPostprocessor


class AdaptiveThresholding(IRetrievalPostprocessor):
    def __init__(self) -> None:
        print("hello")

    def process(self, *args, **kwargs) -> Any:  # type: ignore
        pass


__all__ = ["AdaptiveThresholding"]
