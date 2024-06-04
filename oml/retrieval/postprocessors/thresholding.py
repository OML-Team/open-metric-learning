from oml.interfaces.retrieval import IRetrievalPostprocessor


class AdaptiveThresholding(IRetrievalPostprocessor):
    def __init__(self) -> None:
        print("hello")


__all__ = ["AdaptiveThresholding"]
