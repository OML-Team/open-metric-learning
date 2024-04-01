from pathlib import Path
from typing import Optional, Union

import numpy as np
import PIL
import torch
from PIL.Image import Image as TPILImage
from torch import nn

from oml.const import MEAN, STD, STORAGE_CKPTS, TNormParam
from oml.interfaces.models import IExtractor
from oml.models.utils import (
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.models.vit_dino.external.hubconf import (  # type: ignore
    dino_vitb8,
    dino_vitb16,
    dino_vits8,
    dino_vits16,
)
from oml.models.vit_dino.external_v2.hubconf import (  # type: ignore
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)
from oml.transforms.images.albumentations import get_normalisation_albu
from oml.utils.io import download_checkpoint_one_of
from oml.utils.misc_torch import normalise, temporary_setting_model_mode

_FB_URL = "https://dl.fbaipublicfiles.com"


class ViTExtractor(IExtractor):
    """
    The base class for the extractors that follow VisualTransformer architecture.

    """

    constructors = {
        "vits8": dino_vits8,
        "vits16": dino_vits16,
        "vitb8": dino_vitb8,
        "vitb16": dino_vitb16,
        "vits14": dinov2_vits14,
        "vitb14": dinov2_vitb14,
        "vitl14": dinov2_vitl14,
        "vits14_reg": dinov2_vits14_reg,
        "vitb14_reg": dinov2_vitb14_reg,
        "vitl14_reg": dinov2_vitl14_reg,
    }

    pretrained_models = {
        # checkpoints pretrained in DINO framework on ImageNet by MetaAI
        "vits16_dino": {
            "url": f"{_FB_URL}/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            "hash": "cf0f22",
            "fname": "vits16_dino.ckpt",
            "init_args": {"arch": "vits16", "normalise_features": False},
        },
        "vits8_dino": {
            "url": f"{_FB_URL}/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            "hash": "230cd5",
            "fname": "vits8_dino.ckpt",
            "init_args": {"arch": "vits8", "normalise_features": False},
        },
        "vitb16_dino": {
            "url": f"{_FB_URL}/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            "hash": "552daf",
            "fname": "vitb16_dino.ckpt",
            "init_args": {"arch": "vitb16", "normalise_features": False},
        },
        "vitb8_dino": {
            "url": f"{_FB_URL}/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            "hash": "556550",
            "fname": "vitb8_dino.ckpt",
            "init_args": {"arch": "vitb8", "normalise_features": False},
        },
        "vits14_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "hash": "2e405c",
            "fname": "dinov2_vits14.ckpt",
            "init_args": {"arch": "vits14", "normalise_features": False},
        },
        "vits14_reg_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
            "hash": "2a50c5",
            "fname": "dinov2_vits14_reg4.ckpt",
            "init_args": {"arch": "vits14_reg", "normalise_features": False},
        },
        "vitb14_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "hash": "8635e7",
            "fname": "dinov2_vitb14.ckpt",
            "init_args": {"arch": "vitb14", "normalise_features": False},
        },
        "vitb14_reg_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
            "hash": "13d13c",
            "fname": "dinov2_vitb14_reg4.ckpt",
            "init_args": {"arch": "vitb14_reg", "normalise_features": False},
        },
        "vitl14_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "hash": "19a02c",
            "fname": "dinov2_vitl14.ckpt",
            "init_args": {"arch": "vitl14", "normalise_features": False},
        },
        "vitl14_reg_dinov2": {
            "url": f"{_FB_URL}/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
            "hash": "8b6364",
            "fname": "dinov2_vitl14_reg4.ckpt",
            "init_args": {"arch": "vitl14_reg", "normalise_features": False},
        },
        # our pretrained checkpoints
        "vits16_inshop": {
            "url": [
                f"{STORAGE_CKPTS}/inshop/vits16_inshop_a76b85.ckpt",
                "1niX-TC8cj6j369t7iU2baHQSVN3MVJbW",
            ],
            "hash": "a76b85",
            "fname": "vits16_inshop.ckpt",
            "init_args": {"arch": "vits16", "normalise_features": False},
        },
        "vits16_sop": {
            "url": [
                f"{STORAGE_CKPTS}/sop/vits16_sop_21e743.ckpt",
                "1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A",
            ],
            "hash": "21e743",
            "fname": "vits16_sop.ckpt",
            "init_args": {"arch": "vits16", "normalise_features": True},
        },
        "vits16_cub": {
            "url": [
                f"{STORAGE_CKPTS}/cub/vits16_cub.ckpt",
                "1p2tUosFpGXh5sCCdzlXtjV87kCDfG34G",
            ],
            "hash": "e82633",
            "fname": "vits16_cub.ckpt",
            "init_args": {"arch": "vits16", "normalise_features": False},
        },
        "vits16_cars": {
            "url": [
                f"{STORAGE_CKPTS}/cars/vits16_cars.ckpt",
                "1hcOxDRRXrKr6ZTCyBauaY8Ue-pok4Icg",
            ],
            "hash": "9f1e59",
            "fname": "vits16_cars.ckpt",
            "init_args": {"arch": "vits16", "normalise_features": False},
        },
    }

    def __init__(
        self,
        weights: Optional[Union[Path, str]],
        arch: str,
        normalise_features: bool,
        use_multi_scale: bool = False,
    ):
        """
        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to
             randomly initialize model's weights. You can check the available pretrained checkpoints
             in ``self.pretrained_models``.
            arch: Might be one of ``vits8``, ``vits16``, ``vitb8``, ``vitb16``. You can check all the available options
             in ``self.constructors``
            normalise_features: Set ``True`` to normalise output features
            use_multi_scale: Set ``True`` to use multiscale (the analogue of test time augmentations)

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
            pretrained = self.pretrained_models[weights]  # type: ignore
            weights = download_checkpoint_one_of(
                url_or_fid_list=pretrained["url"],  # type: ignore
                hash_md5=pretrained["hash"],  # type: ignore
                fname=pretrained["fname"],  # type: ignore
            )

        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state_dict = remove_criterion_in_state_dict(state_dict)
        ckpt = remove_prefix_from_state_dict(state_dict, trial_key="norm.bias")
        self.model.load_state_dict(ckpt, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mscale:
            x = self._multi_scale(x)
        else:
            x = self.model(x)

        if self.normalise_features:
            x = normalise(x)

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

    def draw_attention(self, image: Union[TPILImage, np.ndarray]) -> np.ndarray:
        """
        Args:
            image: An image with pixel values in the range of ``[0..255]``.

        Returns:
            An image with drawn attention maps.

        Visualization of the multi-head attention on a particular image.

        """
        return vis_vit(vit=self, image=image)


def vis_vit(
    vit: ViTExtractor,
    image: Union[TPILImage, np.ndarray],
    mean: TNormParam = MEAN,
    std: TNormParam = STD,
) -> np.ndarray:
    # this is the optional dependency
    from pytorch_grad_cam.utils.image import show_cam_on_image

    need_to_convert = not isinstance(image, np.ndarray)

    if need_to_convert:
        image = np.asarray(image)

    patch_size = vit.model.patch_embed.proj.kernel_size[0]

    img_tensor = get_normalisation_albu(mean=mean, std=std)(image=image)["image"]

    w = img_tensor.shape[1] - img_tensor.shape[1] % patch_size
    h = img_tensor.shape[2] - img_tensor.shape[2] % patch_size

    img_tensor = img_tensor[:, :w, :h].unsqueeze(0)

    w_feat_map = img_tensor.shape[-2] // patch_size
    h_feat_map = img_tensor.shape[-1] // patch_size

    with temporary_setting_model_mode(vit, set_train=False):
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

    if need_to_convert:
        arr = PIL.Image.fromarray(arr)

    return arr


__all__ = ["ViTExtractor", "vis_vit"]
