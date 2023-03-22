import hydra
from omegaconf import DictConfig

from oml.lightning.pipelines.validate import extractor_validation_pipeline


@hydra.main(config_path="", config_name="val_sop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    extractor_validation_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
