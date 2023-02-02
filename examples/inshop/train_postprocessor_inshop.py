import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train_postprocessor import (
    pl_train_postprocessor,  # type: ignore
)


@hydra.main(config_path="configs_experimental", config_name="train_postprocessor_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()
