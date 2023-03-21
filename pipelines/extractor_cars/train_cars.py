import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import extractor_training_pipeline


@hydra.main(config_path=".", config_name="train_cars.yaml")
def main_hydra(cfg: DictConfig) -> None:
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
