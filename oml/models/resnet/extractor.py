from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image
import torch
from PIL.Image import Image as TPILImage
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from oml.interfaces.models import IExtractor
from oml.models.resnet.pooling import GEM
from oml.models.utils import (
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.transforms.images.albumentations import get_normalisation_albu
from oml.utils.io import download_checkpoint
from oml.utils.misc_torch import get_device, normalise


def resnet50_projector() -> nn.Module:
    model = resnet50(weights=None, num_classes=128)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.weight.shape[1], model.fc.weight.shape[1]),
        nn.ReLU(),
        model.fc,
    )
    return model


class ResnetExtractor(IExtractor):
    """
    The base class for the extractors that follow ResNet architecture.

    """

    constructors = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet50_projector": resnet50_projector,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    pretrained_models = {
        "resnet50_moco_v2": {
            "url": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
            "hash": "a04e12f8",
            "fname": "moco_v2_800ep_pretrain.pth.tar",
            "init_args": {"arch": "resnet50_projector", "remove_fc": True, "normalise_features": False, "gem_p": 5.0},
        },
        "resnet18_imagenet1k_v1": {
            "init_args": {"arch": "resnet18", "remove_fc": True, "normalise_features": False, "gem_p": None}
        },
        "resnet34_imagenet1k_v1": {
            "init_args": {"arch": "resnet34", "remove_fc": True, "normalise_features": False, "gem_p": None}
        },
        "resnet50_imagenet1k_v1": {
            "init_args": {"arch": "resnet50", "remove_fc": True, "normalise_features": False, "gem_p": None},
        },
        "resnet101_imagenet1k_v1": {
            "init_args": {"arch": "resnet101", "remove_fc": True, "normalise_features": False, "gem_p": None},
        },
        "resnet152_imagenet1k_v1": {
            "init_args": {"arch": "resnet152", "remove_fc": True, "normalise_features": False, "gem_p": None},
        },
    }

    def __init__(
        self,
        weights: Optional[Union[Path, str]],
        arch: str,
        gem_p: Optional[float],
        remove_fc: bool,
        normalise_features: bool,
    ):
        """

        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to randomly
             initialize model's weights. You can check the available pretrained checkpoints
             in ``self.pretrained_models``.
            arch: Different types of ResNet, please, check ``self.constructors``
            gem_p: Value of power in `Generalized Mean Pooling` that we use as the replacement for the default one
             (if ``gem_p == 1`` or ``None`` it's just a normal average pooling and if ``gem_p -> inf`` it's max-pooling)
            remove_fc: Set ``True`` if you want to remove the last fully connected layer. Note, that having this layer
              is obligatory for calling ``draw_gradcam()`` method
            normalise_features: Set ``True`` to normalise output features

        """
        assert arch in self.constructors.keys()
        super(ResnetExtractor, self).__init__()

        self.arch = arch
        self.gem_p = gem_p
        self.remove_fc = remove_fc
        self.normalise_features = normalise_features

        constructor = self.constructors[self.arch]
        self.model = constructor()

        if gem_p is not None:
            self.model.avgpool = GEM(p=gem_p)

        if weights is None:
            if self.remove_fc:
                self.model.fc = nn.Identity()
            return

        elif weights == "resnet50_moco_v2":
            pretrained = self.pretrained_models[weights]  # type: ignore
            moco_path = download_checkpoint(
                url_or_fid=pretrained["url"],  # type: ignore
                hash_md5=pretrained["hash"],  # type: ignore
                fname=pretrained["fname"],  # type: ignore
            )
            state_dict = load_moco_weights(moco_path)

        elif str(weights).endswith("_imagenet1k_v1"):
            state_dict = constructor(weights="IMAGENET1K_V1").state_dict()

        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        state_dict = remove_criterion_in_state_dict(state_dict)  # type: ignore
        state_dict = remove_prefix_from_state_dict(state_dict, "layer4.")  # type: ignore

        if self.remove_fc:
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            if arch != "resnet50_projector":
                self.model.fc = nn.Identity()

        self.model.load_state_dict(state_dict, strict=True)

        if self.remove_fc:
            self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            x = normalise(x)

        return x

    def get_last_conv_channels(self) -> int:
        last_block = self.model.layer4[-1]
        if self.arch in ("resnet18", "resnet34"):
            n_out_channels = last_block.conv2.out_channels
        else:
            # resnet50, resnet101, resnet152
            n_out_channels = last_block.conv3.out_channels
        return n_out_channels

    @property
    def feat_dim(self) -> int:
        if isinstance(self.model.fc, torch.nn.Identity):
            return self.get_last_conv_channels()
        elif isinstance(self.model.fc, torch.nn.Linear):
            return self.model.fc.out_features
        else:
            # 2-layer mlp case
            return self.model.fc[-1].out_features

    def draw_gradcam(self, image: Union[np.ndarray, TPILImage]) -> Union[np.ndarray, TPILImage]:
        """
        Args:
            image: An image with pixel values in the range of ``[0..255]``.

        Returns:
            An image with drawn gradients.

        Visualization of the gradients on a particular image using `GradCam`_.

        .. _GradCam: https://arxiv.org/abs/1610.02391

        """
        # this is the optional dependency
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        if self.remove_fc:
            raise ValueError("This method does not work if there is no FC layer in the model.")

        need_to_convert = not isinstance(image, np.ndarray)

        if need_to_convert:
            image = np.asarray(image)

        device = get_device(self.model)
        image_tensor = get_normalisation_albu()(image=image)["image"].to(device)
        cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]], use_cuda=device != "cpu")
        gray_image = cam(image_tensor.unsqueeze(0), None)[0]
        img_with_grads = show_cam_on_image(image / 255, gray_image)

        if need_to_convert:
            img_with_grads = PIL.Image.fromarray(img_with_grads)

        return img_with_grads


def load_moco_weights(path_to_model: Union[str, Path]) -> Dict[str, Any]:
    """
    Args:
        path_to_model: Path to model trained using original
           code from MoCo repository:
           https://github.com/facebookresearch/moco

    Returns:
        State dict without weights of student

    """
    checkpoint = torch.load(path_to_model, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    for key in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if key.startswith("module.encoder_q"):
            new_key = key[len("module.encoder_q.") :]
            state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return state_dict


__all__ = ["ResnetExtractor"]
