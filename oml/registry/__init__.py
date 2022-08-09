from oml.registry.losses import LOSSES_REGISTRY
from oml.registry.miners import MINERS_REGISTRY
from oml.registry.models import MODELS_REGISTRY
from oml.registry.optimizers import OPTIMIZERS_REGISTRY
from oml.registry.samplers import SAMPLERS_REGISTRY
from oml.registry.schedulers import SCHEDULERS_REGISTRY
from oml.registry.transforms import AUGS_REGISTRY


def show_registry() -> None:
    for name, registry in [
        ("Losses", LOSSES_REGISTRY),
        ("Miners", MINERS_REGISTRY),
        ("Models", MODELS_REGISTRY),
        ("Optimizers", OPTIMIZERS_REGISTRY),
        ("Samplers", SAMPLERS_REGISTRY),
        ("Schedulers", SCHEDULERS_REGISTRY),
        ("Augmentations", AUGS_REGISTRY),
    ]:
        print(f"{name}: ")
        print(list(registry.keys()))  # type: ignore
        print()
