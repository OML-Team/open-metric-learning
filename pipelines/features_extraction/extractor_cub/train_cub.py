import hydra
from omegaconf import DictConfig

from oml.lightning.pipelines.train import extractor_training_pipeline


@hydra.main(config_path="", config_name="train_cub.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on SOP dataset.")
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
