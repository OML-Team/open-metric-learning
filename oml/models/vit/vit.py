from pathlib import Path
from typing import Union

import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

from oml.interfaces.models import IExtractor
from oml.models.utils import remove_prefix_from_state_dict
from oml.models.vit.hubconf import dino_vitb8  # type: ignore
from oml.models.vit.hubconf import dino_vitb16  # type: ignore
from oml.models.vit.hubconf import dino_vits8  # type: ignore
from oml.models.vit.hubconf import dino_vits16  # type: ignore
from oml.transforms.images.albumentations.shared import get_default_transforms_albu


class ViTExtractor(IExtractor):
    constructors = {"vits8": dino_vits8, "vits16": dino_vits16, "vitb8": dino_vitb8, "vitb16": dino_vitb16}

    def __init__(
        self, weights: Union[Path, str], arch: str, normalise_features: bool, use_multi_scale: bool, strict_load: bool
    ):
        """
        Args:
            weights: path to weights, or use "pretrained_dino" to download pretrained checkpoint
            arch: "vits8", "vits16", "vitb8", "vitb16"; check all of the available options in self.constructor
            normalise_features: if normalise features
            use_multi_scale: if use multi scale
            strict_load: if strict load from checkpoint

        """
        assert arch in self.constructors.keys()
        super(ViTExtractor, self).__init__()

        self.normalise_features = normalise_features
        self.mscale = use_multi_scale
        self.arch = arch

        factory_fun = self.constructors[self.arch]

        if weights == "pretrained_dino":
            self.model = factory_fun(pretrained=True)
        elif weights == "random":
            self.model = factory_fun(pretrained=False)
        else:
            self.model = factory_fun(pretrained=False)
            ckpt = torch.load(weights, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt.keys() else ckpt
            ckpt = remove_prefix_from_state_dict(state_dict, "norm.bias")

            self.model.load_state_dict(ckpt, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mscale:
            x = self.multi_scale(x)
        else:
            x = self.model(x)

        if self.normalise_features:
            xn = torch.linalg.norm(x, 2, dim=1).detach()
            x = x.div(xn.unsqueeze(1))

        return x

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def feat_dim(self) -> int:
        return len(self.model.norm.bias)

    def multi_scale(self, samples: torch.Tensor) -> torch.Tensor:
        return multi_scale(samples=samples, model=self.model)

    def draw_attention(self, image: np.ndarray) -> np.ndarray:
        return vis_vit(vit=self, image=image)


def vis_vit(vit: ViTExtractor, image: np.ndarray) -> np.ndarray:
    """
    Visualisation of multi heads attention.

    Args:
        vit: VIT model
        image: Input image

    Returns:
        Image with attention maps drawn on top of the input image

    """
    vit.eval()

    patch_size = vit.model.patch_embed.proj.kernel_size[0]

    img_tensor = get_default_transforms_albu()(image=image)["image"]

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


def multi_scale(samples: torch.Tensor, model: nn.Module) -> torch.Tensor:
    # code from the original DINO
    v = None
    scales = [1, 1 / 2 ** (1 / 2), 1 / 2]  # we use 3 different scales
    for s in scales:
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode="bilinear", align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= len(scales)
    # v /= v.norm(dim=1)  # we don't want to shift the norms values
    return v
