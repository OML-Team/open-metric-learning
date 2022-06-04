from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torch.nn import functional
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from oml.interfaces.models import IExtractor
from oml.models.pooling import GEM
from oml.models.utils import remove_prefix_from_state_dict
from oml.utils.images.augs import get_default_transforms_albu
from oml.utils.io import download_checkpoint


class ResnetExtractor(IExtractor):
    moco_v2_800_epoch = (
        "https://dl.fbaipublicfiles.com/" "moco/moco_checkpoints/moco_v2_800ep/" "moco_v2_800ep_pretrain.pth.tar",
        "a04e12f8",
    )

    constructors = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    def __init__(
        self,
        weights: Union[Path, str],
        arch: str,
        hid_dim: Optional[int],
        out_dim: Optional[int],
        normalise_features: bool,
        gem_p: Optional[float],
        remove_fc: bool,
        strict_load: bool,
    ):
        """
        If you want to load FC layer from checkpoint and keep it untouched,
        set both hid_dim and out_dim equal to None.

        Args:
            weights: path to weights, or use "pretrained_moco" to download pretrained checkpoint
            arch: different types of resnet, please, check self.constructors
            hid_dim: hidden dimension
            out_dim: output dimension
            normalise_features: if normalise features
            gem_p: value of power in GEM pooling
            remove_fc: if remove fully connected layer
            strict_load:  if strict load from checkpoint

        """
        assert arch in self.constructors.keys()

        super(ResnetExtractor, self).__init__()

        self.arch = arch
        self.normalise_features = normalise_features
        self.remove_fc = remove_fc

        factory_fun = self.constructors[self.arch]

        self.model = factory_fun(pretrained=False)
        self.last_conv_channels = self.calc_last_conv_channels()

        keep_fc_untouched = (hid_dim is None) and (out_dim is None)

        # change last fc layer if needed
        if keep_fc_untouched:
            pass

        elif (hid_dim is not None) and (out_dim is not None):
            head_layers = [
                nn.Linear(in_features=self.last_conv_channels, out_features=hid_dim),
                nn.ReLU(),
                nn.Linear(in_features=hid_dim, out_features=out_dim),
            ]
            if weights != "pretrained_moco":
                # insert bn before relu
                head_layers.insert(1, nn.BatchNorm1d(num_features=hid_dim))

            self.model.fc = nn.Sequential(*head_layers)

        elif (hid_dim is None) and (out_dim is not None):
            self.model.fc = nn.Linear(in_features=self.last_conv_channels, out_features=out_dim)

        else:
            raise ValueError("Unexpected cfg: (hid_dim is not None) and (out_dim is None)")

        if gem_p is not None:
            self.model.avgpool = GEM(p=gem_p)

        if remove_fc:
            assert (hid_dim is None) and (out_dim is None)
            self.model.fc = nn.Identity()

        if weights == "random":
            return
        elif weights == "pretrained":
            state_dict = factory_fun(pretrained=True).state_dict()
        elif weights == "pretrained_moco":
            assert self.arch == "resnet50", "We have MoCo model only for ResNet50"
            url, hash_md5 = self.moco_v2_800_epoch
            path_to_model = download_checkpoint(url=url, hash_md5=hash_md5)
            state_dict = load_moco_model(path_to_model=Path(path_to_model)).state_dict()
        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        state_dict = remove_prefix_from_state_dict(state_dict, "layer4.")

        if not keep_fc_untouched:
            assert not strict_load, "Strict load has to be False if you want to change original FC"
            print("FC layer from the original checkpoint will be removed.")
            fc_keys = [key for key in state_dict.keys() if key.startswith("fc")]
            for key in fc_keys:
                del state_dict[key]

        self.model.load_state_dict(state_dict, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            x = functional.normalize(x)

        return x

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

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

    def draw_attention(self, image: np.ndarray) -> np.ndarray:
        model_device = str(list(self.model.parameters())[0].device)
        image_tensor = get_default_transforms_albu()(image=image)["image"].to(model_device)
        cam = GradCAM(model=self.model, target_layer=self.model.layer4[-1], use_cuda=not (model_device == "cpu"))
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

    model = resnet50(num_classes=128)

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
