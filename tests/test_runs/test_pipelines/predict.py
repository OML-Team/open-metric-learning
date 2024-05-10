import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR, MOCK_DATASET_PATH
from oml.lightning.pipelines.predict import extractor_prediction_pipeline
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="predict.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_mock_dataset(MOCK_DATASET_PATH)
    cfg["data_dir"] = str(MOCK_DATASET_PATH)
    extractor_prediction_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
