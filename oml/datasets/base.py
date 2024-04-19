from oml.datasets.images import ImagesDatasetLabeled, ImagesDatasetQueryGalleryLabeled


class DatasetWithLabels(ImagesDatasetLabeled):
    # this class allows to have back compatibility
    pass


class DatasetQueryGallery(ImagesDatasetQueryGalleryLabeled):
    # this class allows to have back compatibility
    pass


__all__ = ["DatasetWithLabels", "DatasetQueryGallery"]
