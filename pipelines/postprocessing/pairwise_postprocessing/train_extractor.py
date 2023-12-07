import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR
from oml.lightning.pipelines.train import extractor_training_pipeline


@hydra.main(config_path=".", config_name="extractor_train.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
