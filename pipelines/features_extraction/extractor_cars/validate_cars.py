import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_VERSION
from oml.lightning.pipelines.validate import extractor_validation_pipeline


@hydra.main(config_path=".", config_name="val_cars.yaml", version_base=HYDRA_VERSION)
def main_hydra(cfg: DictConfig) -> None:
    extractor_validation_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
