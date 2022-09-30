import inspect

from oml.registry.losses import LOSSES_REGISTRY
from oml.registry.miners import MINERS_REGISTRY
from oml.registry.models import MODELS_REGISTRY
from oml.registry.optimizers import OPTIMIZERS_REGISTRY
from oml.registry.samplers import SAMPLERS_REGISTRY
from oml.registry.schedulers import SCHEDULERS_REGISTRY
from oml.registry.transforms import TRANSFORMS_REGISTRY


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
