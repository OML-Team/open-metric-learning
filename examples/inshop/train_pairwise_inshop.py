import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train_pairwise import main  # type: ignore


@hydra.main(config_path="configs", config_name="train_pairwise_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_hydra()
