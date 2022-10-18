import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import pl_train_return_main_metric


@hydra.main(config_path="configs", config_name="train_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    return pl_train_return_main_metric(cfg)


if __name__ == "__main__":
    main_hydra()
