import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.validate import main


@hydra.main(config_path="configs", config_name="val_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_hydra()
