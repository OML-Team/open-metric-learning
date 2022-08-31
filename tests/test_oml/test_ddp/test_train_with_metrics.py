import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import pytest
import torch

from oml.const import OVERALL_CATEGORIES_KEY, PROJECT_ROOT, TMP_PATH


@pytest.mark.parametrize(
    "devices",
    [
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize("batch_size", [7, 19])
@pytest.mark.parametrize("num_labels,atol", [(200, 1e-2), (1000, 2e-3)])
def test_metrics_is_similar_in_ddp(devices: Tuple[int, ...], num_labels: int, atol: float, batch_size: int) -> None:
    # We will compare metrics from same experiment but with different amount of devices. For this we aggregate
    # metrics in variable with following structure:
    # {"cmc_<K>": [cmc_<K>_<exp_1>, cmc_<K>_<exp_2>, cmc_<K>_<exp_3>, ...],
    #  "map_<K>": [map_<K>_<exp_1>, map_<K>_<exp_2>, map_<K>_<exp_3>, ...]}
    # After that we will compare metrics. Metrics shouldn't be equal, but very similar
    metric_topk2values = defaultdict(list)

    exp_file = PROJECT_ROOT / "tests/test_oml/test_ddp/new_exp.py"

    for num_devices in devices:
        save_path = TMP_PATH / f"devices_{num_devices}_{num_labels}.pth"
        params = (
            f"--devices {num_devices} "
            f"--save_path {str(save_path)} "
            f"--num_labels {num_labels} "
            f"--batch_size {batch_size}"
        )
        cmd = f"python {exp_file} " + params
        subprocess.run(cmd, check=True, shell=True)

        metrics = torch.load(save_path)[OVERALL_CATEGORIES_KEY]
        # save_path.unlink(missing_ok=True)
        for metric_name, topk2value in metrics.items():
            for top_k, value in topk2value.items():
                metric_topk2values[f"{metric_name}_{top_k}"].append(value)

    compare_metrics(metric_topk2values, atol)


def compare_metrics(metric_topk2values: Dict[str, List[torch.Tensor]], atol: float) -> None:
    for metric_topk, values in metric_topk2values.items():
        mean_value = torch.tensor(values).float().mean()
        is_close = tuple(torch.isclose(val, mean_value, atol=atol) for val in values)
        assert all(is_close), f"Metrics are not similar: {[metric_topk, values, is_close, atol]}"
