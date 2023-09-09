import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_VERSION, MOCK_DATASET_PATH
from oml.lightning.pipelines.validate import extractor_validation_pipeline
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="valid_arc_dino.yaml", version_base=HYDRA_VERSION)
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_mock_dataset(MOCK_DATASET_PATH)
    cfg["dataset_root"] = MOCK_DATASET_PATH
    extractor_validation_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
