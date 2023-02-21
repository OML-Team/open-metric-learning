from pathlib import Path

import torch

path_to_ckpt = Path("/nydl/logs/8feb/InShop/2023-02-15_12-12-36_postprocessing_no_categories/checkpoints/best.ckpt")

y = torch.load(path_to_ckpt, map_location="cpu")["state_dict"]

kvs = list(y.items())
for k, v in kvs:
    if k.startswith("model_pairwise"):
        del y[k]

path_to_save = path_to_ckpt.parent / "best_fixed.ckpt"
torch.save({"state_dict": y}, path_to_save)
