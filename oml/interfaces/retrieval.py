from typing import Any


class IRetrievalPostprocessor:
    """
    This is a base interface for the classes which somehow postprocess retrieval results.

    """

    def process(self, *args, **kwargs) -> Any:  # type: ignore
        # todo 522: add actual signature later
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
