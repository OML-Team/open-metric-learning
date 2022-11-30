import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import pl_train


@hydra.main(config_path="configs", config_name="train_sop_surrogate_precision.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on SOP dataset.")
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
