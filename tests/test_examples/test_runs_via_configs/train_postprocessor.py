from pathlib import Path

import hydra
from omegaconf import DictConfig

from oml.const import MOCK_DATASET_PATH
from oml.lightning.entrypoints.train_pairwise import pl_train_postprocessor
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="train_postprocessor.yaml")
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)

    # since we will store embedding in the dataset folder, its hash will change
    # in order not to affect other tests, we keep this dataset separately
    download_mock_dataset(Path(cfg["logs_root"]) / "mock_dataset_with_embeddings", check_md5=False)

    cfg["dataset_root"] = MOCK_DATASET_PATH
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()
