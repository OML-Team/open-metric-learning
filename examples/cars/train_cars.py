import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import main


@hydra.main(config_path="configs", config_name="train_cars.yaml")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_hydra()
