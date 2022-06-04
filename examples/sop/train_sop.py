import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import main


@hydra.main(config_path="configs", config_name="train_sop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on SOP dataset.")
    main(cfg)


if __name__ == "__main__":
    main_hydra()
