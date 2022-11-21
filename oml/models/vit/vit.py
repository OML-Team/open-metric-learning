from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

from oml.const import MEAN, STD, STORAGE_CKPTS, TNormParam
from oml.interfaces.models import IExtractor
from oml.models.utils import remove_prefix_from_state_dict
from oml.models.vit.hubconf import (  # type: ignore
    dino_vitb8,
    dino_vitb16,
    dino_vits8,
    dino_vits16,
)
from oml.transforms.images.albumentations.transforms import get_normalisation_albu
from oml.utils.io import download_checkpoint_one_of

_FB_URL = "https://dl.fbaipublicfiles.com"


class ViTExtractor(IExtractor):
    """
    The base class for the extractors that follow VisualTransformer architecture.

    """

    constructors = {"vits8": dino_vits8, "vits16": dino_vits16, "vitb8": dino_vitb8, "vitb16": dino_vitb16}

    pretrained_models = {
        # checkpoints pretrained in DINO framework on ImageNet by MetaAI
        "vits16_dino": (f"{_FB_URL}/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", "cf0f22", None),
        "vits8_dino": (f"{_FB_URL}/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth", "230cd5", None),
        "vitb16_dino": (f"{_FB_URL}/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth", "552daf", None),
        "vitb8_dino": (f"{_FB_URL}/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", "556550", None),
        # our pretrained checkpoints
        "vits16_inshop": (
            [f"{STORAGE_CKPTS}/inshop/vits16_inshop_a76b85.ckpt", "1niX-TC8cj6j369t7iU2baHQSVN3MVJbW"],
            "a76b85",
            "vits16_inshop.ckpt",
        ),
        "vits16_sop": (
            [f"{STORAGE_CKPTS}/sop/vits16_sop_21e743.ckpt", "1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A"],
            "21e743",
            "vits16_sop.ckpt",
        ),
        "vits16_cub": (
            [f"{STORAGE_CKPTS}/cub/vits16_cub.ckpt", "1p2tUosFpGXh5sCCdzlXtjV87kCDfG34G"],
            "e82633",
            "vits16_cub.ckpt",
        ),
        "vits16_cars": (
            [f"{STORAGE_CKPTS}/cars/vits16_cars.ckpt", "1hcOxDRRXrKr6ZTCyBauaY8Ue-pok4Icg"],
            "9f1e59",
            "vits16_cars.ckpt",
        ),
    }

    def __init__(
        self,
        weights: Optional[Union[Path, str]],
        arch: str,
        normalise_features: bool,
        use_multi_scale: bool = False,
        strict_load: bool = True,
    ):
        """
        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to
             randomly initialize model's weights. You can check the available pretrained checkpoints
             in ``self.pretrained_models``.
            arch: Might be one of ``vits8``, ``vits16``, ``vitb8``, ``vitb16``. You can check all the available options
             in ``self.constructors``
            normalise_features: Set ``True`` to normalise output features
            use_multi_scale: Set ``True`` to use multi scale (the analogue of test time augmentations)
            strict_load: Set ``True`` if you want the strict load of the weights from the checkpoint

        """
        assert arch in self.constructors
        super(ViTExtractor, self).__init__()

        self.normalise_features = normalise_features
        self.mscale = use_multi_scale
        self.arch = arch

        factory_fun = self.constructors[self.arch]

        self.model = factory_fun(pretrained=False)
        if weights is None:
            return

        if weights in self.pretrained_models:
            url_or_fid, hash_md5, fname = self.pretrained_models[weights]  # type: ignore
            weights = download_checkpoint_one_of(
                url_or_fid_list=url_or_fid, hash_md5=hash_md5, fname=fname  # type: ignore
            )

        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        ckpt = remove_prefix_from_state_dict(state_dict, trial_key="norm.bias")
        self.model.load_state_dict(ckpt, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mscale:
            x = self._multi_scale(x)
        else:
            x = self.model(x)

        if self.normalise_features:
            xn = torch.linalg.norm(x, 2, dim=1).detach()
            x = x.div(xn.unsqueeze(1))

        return x

    @property
    def feat_dim(self) -> int:
        return len(self.model.norm.bias)

    def _multi_scale(self, samples: torch.Tensor) -> torch.Tensor:
        # code from the original DINO
        # TODO: check grads later
        v = torch.zeros((len(samples), self.feat_dim), device=samples.device)
        scales = [1.0, 1 / 2 ** (1 / 2), 1 / 2]  # we use 3 different scales
        for s in scales:
            if s == 1:
                inp = samples.clone()
            else:
                inp = nn.functional.interpolate(samples, scale_factor=s, mode="bilinear", align_corners=False)
            feats = self.model.forward(inp).clone()
            v += feats

        v /= len(scales)
        # v /= v.norm(dim=1)  # we don't want to shift the norms values
        return v

    def draw_attention(self, image: np.ndarray) -> np.ndarray:
        """
        Visualization of the multi-head attention on a particular image.

        """
        return vis_vit(vit=self, image=image)


def vis_vit(vit: ViTExtractor, image: np.ndarray, mean: TNormParam = MEAN, std: TNormParam = STD) -> np.ndarray:
    vit.eval()

    patch_size = vit.model.patch_embed.proj.kernel_size[0]

    img_tensor = get_normalisation_albu(mean=mean, std=std)(image=image)["image"]

    w = img_tensor.shape[1] - img_tensor.shape[1] % patch_size
    h = img_tensor.shape[2] - img_tensor.shape[2] % patch_size

    img_tensor = img_tensor[:, :w, :h].unsqueeze(0)

    w_feat_map = img_tensor.shape[-2] // patch_size
    h_feat_map = img_tensor.shape[-1] // patch_size

    with torch.no_grad():
        attentions = vit.model.get_last_selfattention(img_tensor)

    nh = attentions.shape[1]

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_feat_map, h_feat_map)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .numpy()
    )

    arr = sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0]))

    arr = show_cam_on_image(image / image.max(), 0.6 * arr / arr.max())  # type: ignore

    return arr


__all__ = ["ViTExtractor", "vis_vit"]
