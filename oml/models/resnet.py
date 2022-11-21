from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from oml.interfaces.models import IExtractor
from oml.models.pooling import GEM
from oml.models.utils import remove_prefix_from_state_dict
from oml.transforms.images.albumentations.transforms import get_normalisation_albu
from oml.utils.io import download_checkpoint


class ResnetExtractor(IExtractor):
    """
    The base class for the extractors that follow ResNet architecture.

    """

    constructors = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    pretrained_models = {
        "resnet50_moco_v2": (
            "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
            "a04e12f8",
            None,
        )
    }

    def __init__(
        self,
        weights: Optional[Union[Path, str]],
        arch: str,
        normalise_features: bool,
        gem_p: Optional[float],
        remove_fc: bool,
        strict_load: bool = True,
    ):
        """

        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to randomly
             initialize model's weights or ``default`` to use the checkpoint pretrained on ImageNet.
             You can check the available pretrained checkpoints in ``self.pretrained_models``.
            arch: Different types of ResNet, please, check ``self.constructors``
            normalise_features: Set ``True`` to normalise output features
            gem_p: Value of power in `Generalized Mean Pooling` that we use as the replacement for the default one
             (if ``gem_p == 1`` it's just a normal average pooling and if ``gem_p -> inf`` it's max-pooling)
            remove_fc: Set ``True`` if you want to remove the last fully connected layer
            strict_load: Set ``True`` if you want the strict load of the weights from the checkpoint

        """
        assert arch in self.constructors.keys()

        super(ResnetExtractor, self).__init__()

        self.arch = arch
        self.normalise_features = normalise_features
        self.remove_fc = remove_fc

        factory_fun = self.constructors[self.arch]

        self.model = factory_fun(weights=None)

        if weights == "resnet50_moco_v2":
            # todo:
            # in the next iterations we should use head as a separate entity to change fc
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.weight.shape[1], self.model.fc.weight.shape[1]),
                nn.ReLU(),
                nn.Linear(self.model.fc.weight.shape[1], 128),  # output size in moco v2
            )

        self.last_conv_channels = self.calc_last_conv_channels()

        if gem_p is not None:
            self.model.avgpool = GEM(p=gem_p)

        if weights is None:
            return

        elif isinstance(weights, str) and weights.lower() == "default":
            state_dict = factory_fun(weights="DEFAULT").state_dict()

        elif isinstance(weights, str) and weights.lower() == "imagenet_v1":
            state_dict = factory_fun(weights="IMAGENET1K_V1").state_dict()

        elif weights in self.pretrained_models:
            url_or_fid, hash_md5, fname = self.pretrained_models[weights]  # type: ignore
            path_to_ckpt = download_checkpoint(url_or_fid=url_or_fid, hash_md5=hash_md5, fname=fname)
            state_dict = load_moco_model(path_to_model=Path(path_to_ckpt)).state_dict()

        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        state_dict = remove_prefix_from_state_dict(state_dict, "layer4.")
        self.model.load_state_dict(state_dict, strict=strict_load)

        if remove_fc:
            self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            xn = torch.linalg.norm(x, 2, dim=1).detach()
            x = x.div(xn.unsqueeze(1))

        return x

    def calc_last_conv_channels(self) -> int:
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
            return self.last_conv_channels
        elif isinstance(self.model.fc, torch.nn.Linear):
            return self.model.fc.out_features
        else:
            # 2-layer mlp case
            return self.model.fc[-1].out_features

    def draw_gradcam(self, image: np.ndarray) -> np.ndarray:
        """
        Visualization of the gradients on a particular image using `GradCam`_.

        .. _GradCam: https://arxiv.org/abs/1610.02391

        """
        model_device = str(list(self.model.parameters())[0].device)
        image_tensor = get_normalisation_albu()(image=image)["image"].to(model_device)
        cam = GradCAM(model=self.model, target_layer=self.model.layer4[-1], use_cuda=model_device != "cpu")
        gray_image = cam(image_tensor.unsqueeze(0), "gradcam", None)
        img_with_grads = show_cam_on_image(image / 255, gray_image)
        return img_with_grads


def load_moco_model(path_to_model: Path) -> nn.Module:
    """
    Args:
        path_to_model: Path to model trained using original
           code from MoCo repository:
           https://github.com/facebookresearch/moco

    Returns:
        Model

    """
    checkpoint = torch.load(path_to_model, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q"):
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        del state_dict[k]

    model = resnet50(num_classes=128)  # output size in moco v2

    if "fc.2.weight" in state_dict.keys():
        print("MOCO V2 architecture was detected!")
        model.fc = nn.Sequential(
            nn.Linear(model.fc.weight.shape[1], model.fc.weight.shape[1]),
            nn.ReLU(),
            model.fc,
        )
    else:
        print("MOCO V1 architecture will be used")

    model.load_state_dict(state_dict)

    return model


__all__ = ["ResnetExtractor", "load_moco_model"]
