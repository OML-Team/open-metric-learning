import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

from oml.const import PROJECT_ROOT

from .run_triplets_experiment_ddp import DummyModule

"""
MOTIVATION

With this experiment we want to:
1) Test patching of default dataloaders with `shuffle=False` and `shuffle=True`.
2) Test equality of models with different combinations of parameters `B = sum(Bi * N)`. Where `B` - batch size per
step which is equal for each experiment, `N` - number of devices,
`Bi` - batch size per device.
3) Loaders return the same collections of ids on each step on several epochs.

"""


# TODO: check internal `/` on widnows
exp_file = PROJECT_ROOT / "tests/test_examples/test_ddp_cases/run_triplets_experiment_ddp.py"


def test_epochs_are_equal() -> None:
    max_epochs = 2
    models = {}
    train_ids = defaultdict(list)
    val_ids = defaultdict(list)

    for experiment, (devices, len_dataset, batch_size) in enumerate(
        [(1, 120, 30), (2, 120, 15), (3, 120, 10), (2, 120, 20)]
    ):
        params = (
            f"--devices {devices}",
            f"--len_dataset {len_dataset}",
            f"--exp_num {experiment}",
            f"--batch_size {batch_size}",
            f"--max_epochs {max_epochs}",
        )
        params = " ".join(params)

        cmd = f"python {exp_file} " + params
        subprocess.run(cmd, check=True, shell=True)

        ckpt_path = DummyModule.save_path_ckpt_pattern.format(experiment=experiment)
        models[experiment] = torch.load(ckpt_path, map_location="cpu").requires_grad_(False)
        Path(ckpt_path).unlink(missing_ok=True)

        for epoch in range(max_epochs):
            train_ids_path = DummyModule.save_path_train_ids_pattern.format(experiment=experiment, epoch=epoch)
            train_ids[experiment].append(torch.load(train_ids_path))
            Path(train_ids_path).unlink(missing_ok=True)

            val_ids_path = DummyModule.save_path_val_ids_pattern.format(experiment=experiment, epoch=epoch)
            val_ids[experiment].append(torch.load(val_ids_path))
            Path(val_ids_path).unlink(missing_ok=True)

    assert train_ids[0] == train_ids[1]
    assert train_ids[0] == train_ids[2]
    assert train_ids[0] != train_ids[3]

    assert val_ids[0] == val_ids[1]
    assert val_ids[0] == val_ids[2]
    assert val_ids[0] != val_ids[3]

    eq_01, not_eq_tensors_01 = is_equal_models(models[0], models[1])
    eq_02, not_eq_tensors_02 = is_equal_models(models[0], models[2])
    eq_03, _ = is_equal_models(models[0], models[3])

    assert eq_01, not_eq_tensors_01
    assert eq_02, not_eq_tensors_02
    assert not eq_03


TORCH_EPS = 10 * torch.finfo(torch.float32).eps


def is_equal_models(model1: nn.Module, model2: nn.Module) -> Tuple[bool, List[torch.Tensor]]:
    for module1, module2 in zip(model1.modules(), model2.modules()):
        assert type(module1) == type(module2)
        if isinstance(module1, nn.Linear):
            if not torch.all(torch.isclose(module1.weight, module2.weight, atol=TORCH_EPS)):
                return False, [module1.weight, module2.weight]

    return True, []
