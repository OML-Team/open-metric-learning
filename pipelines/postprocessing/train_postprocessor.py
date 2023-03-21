import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train_postprocessor import (
    pl_train_postprocessor,  # type: ignore
)


@hydra.main(config_path=".", config_name="postprocessor_train.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()
