import hydra
from omegaconf import DictConfig

from oml.const import MOCK_DATASET_PATH
from oml.lightning.entrypoints.train_postprocessor import pl_train_postprocessor
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="train_postprocessor.yaml")
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_mock_dataset(MOCK_DATASET_PATH)
    cfg["dataset_root"] = MOCK_DATASET_PATH
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()
