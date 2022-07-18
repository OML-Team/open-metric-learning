from torchvision.transforms import Compose, Normalize, ToTensor

from oml.const import MEAN, STD, TNormParam


def get_normalisation_torch(mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])
