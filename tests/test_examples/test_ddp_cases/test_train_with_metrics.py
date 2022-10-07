import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pytest
import torch
from torch.utils.data import SequentialSampler

from oml.const import OVERALL_CATEGORIES_KEY, PROJECT_ROOT
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.samplers.balance import BalanceSampler

from .run_retrieval_experiment_ddp import MetricValCallbackWithSaving

rf"""
MOTIVATION

With this experiment, we want to test the patching of loaders with {ModuleDDP} and the similarity of metrics
in DDP mode.

We check the following:
1) Train and Val loaders are split into several parts. These parts have no overlapping except for several samples (for
validation with default {SequentialSampler}) or several batches (for training with {BalanceSampler}) which is necessary
for padding according to the number of devices.
2) Metrics obtained with a different number of devices are very similar. For this purpose, we save metrics and
compare them later. Note that only the final metric should be similar.

Our dummy data is presented by GT labels and PRED labels with some errors. Amount of errors is the same for each
runnings.
"""


exp_file = PROJECT_ROOT / "tests/test_examples/test_ddp_cases/run_retrieval_experiment_ddp.py"


@pytest.mark.parametrize("batch_size", [10, 19])
@pytest.mark.parametrize("max_epochs", [2])
@pytest.mark.parametrize("num_labels,atol", [(200, 5e-3), (1000, 2e-2)])
def test_metrics_is_similar_in_ddp(num_labels: int, atol: float, batch_size: int, max_epochs: int) -> None:
    devices = (1, 2, 3)
    # We will compare metrics from same experiment but with different amount of devices. For this we aggregate
    # metrics in variable with following structure:
    # {"cmc_<K>": [cmc_<K>_<exp_1>, cmc_<K>_<exp_2>, cmc_<K>_<exp_3>, ...],
    #  "map_<K>": [map_<K>_<exp_1>, map_<K>_<exp_2>, map_<K>_<exp_3>, ...]}
    # After that we will compare metrics. Metrics shouldn't be equal, but very similar
    metric_topk2values = defaultdict(list)

    for num_devices in devices:
        params = (
            f"--devices {num_devices} "
            f"--max_epochs {max_epochs} "
            f"--num_labels {num_labels} "
            f"--batch_size {batch_size}"
        )
        cmd = f"python {exp_file} " + params
        subprocess.run(cmd, check=True, shell=True)

        metrics_path = MetricValCallbackWithSaving.save_path_pattern.format(
            devices=num_devices, batch_size=batch_size, num_labels=num_labels
        )
        metrics = torch.load(metrics_path)[OVERALL_CATEGORIES_KEY]
        Path(metrics_path).unlink(missing_ok=True)
        for metric_name, topk2value in metrics.items():
            for top_k, value in topk2value.items():
                metric_topk2values[f"{metric_name}_{top_k}"].append(value)

    compare_metrics(metric_topk2values, atol)


def compare_metrics(metric_topk2values: Dict[str, List[torch.Tensor]], atol: float) -> None:
    # Check point 2 of motivation
    for metric_topk, values in metric_topk2values.items():
        mean_value = torch.tensor(values).float().mean()
        is_close = tuple(torch.isclose(val, mean_value, atol=atol) for val in values)
        assert all(is_close), f"Metrics are not similar: {[metric_topk, values, is_close, atol]}"
