import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR
from oml.lightning.pipelines.train_postprocessor import (
    postprocessor_training_pipeline,  # type: ignore
)


@hydra.main(config_path=".", config_name="postprocessor_train.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    postprocessor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
