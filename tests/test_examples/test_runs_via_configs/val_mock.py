import tempfile

import hydra
from omegaconf import DictConfig

from oml.const import MOCK_DATASET_PATH
from oml.lightning.entrypoints.validate import pl_val
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="val_mock.yaml")
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    cfg["dataset_root"] = MOCK_DATASET_PATH
    cfg["logs_root"] = tempfile.gettempdir()
    pl_val(cfg)


if __name__ == "__main__":
    main_hydra()
