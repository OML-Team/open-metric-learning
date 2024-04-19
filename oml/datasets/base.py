from oml.datasets.images import ImagesDatasetQueryGallery, ImagesDatasetWithLabels


class DatasetWithLabels(ImagesDatasetWithLabels):
    # this class allows to have back compatibility
    pass


class DatasetQueryGallery(ImagesDatasetQueryGallery):
    # this class allows to have back compatibility
    pass


__all__ = ["DatasetWithLabels", "DatasetQueryGallery"]
