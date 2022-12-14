import hydra
from omegaconf import DictConfig

from examples.inference import inference


@hydra.main(config_path="configs", config_name="inf_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    inference(cfg)


if __name__ == "__main__":
    main_hydra()
