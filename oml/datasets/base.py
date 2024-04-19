from oml.datasets.images import ImageDatasetLabeled, ImageDatasetQueryGalleryLabeled


class DatasetWithLabels(ImageDatasetLabeled):
    # this class allows to have backward compatibility
    pass


class DatasetQueryGallery(ImageDatasetQueryGalleryLabeled):
    # this class allows to have backward compatibility
    pass


__all__ = ["DatasetWithLabels", "DatasetQueryGallery"]
