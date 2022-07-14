from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomAffine,
    RandomErasing,
    RandomGrayscale,
    RandomHorizontalFlip,
)


def get_default_torch() -> Compose:
    augs = Compose(
        [RandomHorizontalFlip(), RandomAffine(), GaussianBlur(), ColorJitter(), RandomGrayscale(), RandomErasing()]
    )
    return augs
