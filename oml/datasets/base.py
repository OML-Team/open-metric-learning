from oml.datasets.images import ImageLabeledDataset, ImageQueryGalleryLabeledDataset


class DatasetWithLabels(ImageLabeledDataset):
    # this class allows to have backward compatibility
    pass


class DatasetQueryGallery(ImageQueryGalleryLabeledDataset):
    # this class allows to have backward compatibility
    pass


__all__ = ["DatasetWithLabels", "DatasetQueryGallery"]
