import inspect

from oml.registry.losses import *
from oml.registry.miners import *
from oml.registry.models import *
from oml.registry.optimizers import *
from oml.registry.samplers import *
from oml.registry.schedulers import *
from oml.registry.transforms import *


def show_registry() -> None:
    for name, registry in [
        ("Losses", LOSSES_REGISTRY),
        ("Miners", MINERS_REGISTRY),
        ("Models", MODELS_REGISTRY),
        ("Optimizers", OPTIMIZERS_REGISTRY),
        ("Samplers", SAMPLERS_REGISTRY),
        ("Schedulers", SCHEDULERS_REGISTRY),
        ("Augmentations", TRANSFORMS_REGISTRY),
    ]:
        print(f"{name}: ")

        if name == "Augmentations":
            for k, constructor in registry.items():  # type: ignore
                print(f"{k}:", constructor.__class__.__name__ + str(inspect.signature(constructor)))

        else:
            for k, constructor in registry.items():  # type: ignore
                print(f"{k}:", constructor.__name__ + str(inspect.signature(constructor)))

        print()
