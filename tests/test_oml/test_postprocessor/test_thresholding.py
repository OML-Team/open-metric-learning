from oml.retrieval.postprocessors.thresholding import AdaptiveThresholding


def test_adaptive_thresholding() -> None:
    AdaptiveThresholding(th=0.5)
    assert True
