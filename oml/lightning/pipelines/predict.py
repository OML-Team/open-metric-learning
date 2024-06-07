import itertools
import json
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oml.const import IMAGE_EXTENSIONS, TCfg
from oml.datasets.images import ImageBaseDataset
from oml.ddp.utils import get_world_size_safe, is_main_process, sync_dicts_ddp
from oml.lightning.modules.extractor import ExtractorModule
from oml.lightning.pipelines.parser import parse_engine_params_from_config
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.transforms.images.utils import get_im_reader_for_transforms
from oml.utils.images.images import find_broken_images
from oml.utils.misc import dictconfig_to_dict


def extractor_prediction_pipeline(cfg: TCfg) -> None:
    """
    This pipeline allows you to save features extracted by a feature extractor.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``pipelines/features_extraction/README.md``

    """
    cfg = dictconfig_to_dict(cfg)

    pprint(cfg)

    transforms = get_transforms_by_cfg(cfg["transforms_predict"])
    filenames = [list(Path(cfg["data_dir"]).glob(f"**/*.{ext}")) for ext in IMAGE_EXTENSIONS]
    filenames = list(itertools.chain(*filenames))

    if len(filenames) == 0:
        raise RuntimeError(f"There are no images in the provided directory: {cfg['data_dir']}")

    f_imread = get_im_reader_for_transforms(transforms)

    print("Let's check if there are broken images:")
    broken_images = find_broken_images(filenames, f_imread=f_imread)
    if broken_images:
        raise ValueError(f"There are images that cannot be open:\n {broken_images}.")

    dataset = ImageBaseDataset(paths=filenames, transform=transforms, f_imread=f_imread)

    loader = DataLoader(
        dataset=dataset, batch_size=cfg["bs"], num_workers=cfg["num_workers"], shuffle=False, drop_last=False
    )

    extractor = get_extractor_by_cfg(cfg["extractor"])
    pl_model = ExtractorModule(extractor=extractor)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    trainer_engine_params["use_distributed_sampler"] = True
    trainer = pl.Trainer(precision=cfg.get("precision", 32), **trainer_engine_params)
    predictions = trainer.predict(model=pl_model, dataloaders=loader, return_predictions=True)

    paths, embeddings = [], []
    for prediction in predictions:
        paths.extend([filenames[i] for i in prediction[dataset.index_key].tolist()])
        embeddings.extend(prediction[pl_model.embeddings_key].tolist())

    paths = sync_dicts_ddp({"key": list(map(str, paths))}, get_world_size_safe())["key"]
    embeddings = sync_dicts_ddp({"key": embeddings}, get_world_size_safe())["key"]

    if is_main_process():
        data_to_save = dict(zip(paths, embeddings))
        save_path = Path(cfg["save_dir"]) / "predictions.json"

        with open(save_path, "w") as f:
            json.dump(data_to_save, f)

        print(f"{len(paths)} predictions have been saved to {save_path}")


__all__ = ["extractor_prediction_pipeline"]
