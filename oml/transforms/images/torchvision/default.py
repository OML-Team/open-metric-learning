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
        [
            RandomHorizontalFlip(),
            RandomAffine(degrees=0),
            GaussianBlur(kernel_size=3),
            ColorJitter(),
            RandomGrayscale(),
            RandomErasing(),
        ]
    )
    return augs
