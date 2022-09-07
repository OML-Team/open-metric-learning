import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from oml.const import OVERALL_CATEGORIES_KEY, PROJECT_ROOT
from tests.test_oml.test_ddp.experiment_ddp import MetricValCallbackWithSaving

exp_file = PROJECT_ROOT / "tests/test_oml/test_ddp/experiment_ddp.py"


@pytest.mark.parametrize(
    "devices",
    [
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize("batch_size", [10, 19])
@pytest.mark.parametrize("max_epochs", [3])
@pytest.mark.parametrize("num_labels,atol", [(200, 1e-2), (1000, 3e-3)])
def test_metrics_is_similar_in_ddp(
    devices: Tuple[int, ...], num_labels: int, atol: float, batch_size: int, max_epochs: int
) -> None:
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

        for epoch in range(max_epochs):
            metrics_path = MetricValCallbackWithSaving.save_path_pattern.format(
                devices=num_devices, batch_size=batch_size, num_labels=num_labels, epoch=epoch
            )
            metrics = torch.load(metrics_path)[OVERALL_CATEGORIES_KEY]
            # Path(metrics_path).unlink(missing_ok=True)
            for metric_name, topk2value in metrics.items():
                for top_k, value in topk2value.items():
                    metric_topk2values[f"{metric_name}_{top_k}_epoch_{epoch}"].append(value)

    compare_metrics(metric_topk2values, atol)


def compare_metrics(metric_topk2values: Dict[str, List[torch.Tensor]], atol: float) -> None:
    # Check point 2 of motivation
    for metric_topk, values in metric_topk2values.items():
        mean_value = torch.tensor(values).float().mean()
        is_close = tuple(torch.isclose(val, mean_value, atol=atol) for val in values)
        assert all(is_close), f"Metrics are not similar: {[metric_topk, values, is_close, atol]}"
