from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from oml.interfaces.models import IExtractor
from oml.models.utils import (
    TStateDict,
    filter_state_dict,
    patch_device_and_float,
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.models.vit_clip.external.model import VisionTransformer
from oml.utils.io import download_checkpoint

_OPENAI_URL = "https://openaipublic.azureedge.net/clip/models"
_SBER_URL = "https://huggingface.co/sberbank-ai"


def vitb16_224() -> VisionTransformer:
    return VisionTransformer(
        output_dim=512,
        input_resolution=224,
        layers=12,
        width=768,
        patch_size=16,
        heads=12,
    )


def vitb32_224() -> VisionTransformer:
    return VisionTransformer(
        output_dim=512,
        input_resolution=224,
        layers=12,
        width=768,
        patch_size=32,
        heads=12,
    )


def vitl14_224() -> VisionTransformer:
    return VisionTransformer(
        output_dim=768,
        input_resolution=224,
        layers=24,
        width=1024,
        patch_size=14,
        heads=16,
    )


def vitl14_336() -> VisionTransformer:
    return VisionTransformer(
        output_dim=768,
        input_resolution=336,
        layers=24,
        width=1024,
        patch_size=14,
        heads=16,
    )


class ViTCLIPExtractor(IExtractor):
    constructors = {
        "vitb16_224": vitb16_224,
        "vitb32_224": vitb32_224,
        "vitl14_224": vitl14_224,
        "vitl14_336": vitl14_336,
    }

    pretrained_models: Dict[str, Any] = {
        # checkpoints pretrained by OpenAI
        "openai_vitb16_224": {
            "url": f"{_OPENAI_URL}/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
            "hash": "44c3d804ecac03d9545ac1a3adbca3a6",
            "is_jitted": True,
            "fname": "openai_vitb16_224.ckpt",
            "init_args": {"arch": "vitb16_224", "normalise_features": False},
        },
        "openai_vitb32_224": {
            "url": f"{_OPENAI_URL}/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "hash": "3ba34e387b24dfe590eeb1ae6a8a122b",
            "is_jitted": True,
            "fname": "openai_vitb32_224.ckpt",
            "init_args": {"arch": "vitb32_224", "normalise_features": False},
        },
        "openai_vitl14_224": {
            "url": f"{_OPENAI_URL}/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "hash": "096db1af569b284eb76b3881534822d9",
            "is_jitted": True,
            "fname": "openai_vitl14_224.ckpt",
            "init_args": {"arch": "vitl14_224", "normalise_features": False},
        },
        # checkpoints pretrained by SberbankAI
        "sber_vitb16_224": {
            "url": f"{_SBER_URL}/ruclip-vit-base-patch16-224/resolve/main/pytorch_model.bin",
            "hash": "7882e07674d78c674e33cb892a68bbfc",
            "is_jitted": False,
            "fname": "sber_vitb16_224.ckpt",
            "init_args": {"arch": "vitb16_224", "normalise_features": False},
        },
        "sber_vitb32_224": {
            "url": f"{_SBER_URL}/ruclip-vit-base-patch32-224/resolve/main/pytorch_model.bin",
            "hash": "e2c4dab46a3cfa608bdd762973e90d32",
            "is_jitted": False,
            "fname": "sber_vitb32_224.ckpt",
            "init_args": {"arch": "vitb32_224", "normalise_features": False},
        },
        "sber_vitl14_224": {
            "url": f"{_SBER_URL}/ruclip-vit-large-patch14-224/resolve/main/pytorch_model.bin",
            "hash": "9b4a1cd25d15bad4ffd2ba6e34b8a67c",
            "is_jitted": False,
            "fname": "sber_vitl14_224.ckpt",
            "init_args": {"arch": "vitl14_224", "normalise_features": False},
        },
    }

    def __init__(
        self,
        weights: Optional[str],
        arch: str,
        normalise_features: bool = True,
    ):
        """
        Args:
            weights: Path to weights or special key for pretrained ones or ``None`` for random initialization.
             You can check available pretrained checkpoints in ``ViTCLIPExtractor.pretrained_models``.
            arch: Might be one of ``vitb16_224``, ``vitb32_224``, ``vitl14_224``, ``vitl14_336``.
            normalise_features: Set ``True`` to normalise output features
        """

        super().__init__()

        self.normalize = normalise_features
        self.visual = self.constructors[arch]()

        self.input_size = int(arch.split("_")[-1])

        if weights is None:
            return
        if weights in self.pretrained_models:
            pretrained = self.pretrained_models[weights]
            jitted_weights = pretrained["is_jitted"]
            weights = download_checkpoint(pretrained["url"], pretrained["hash"], fname=pretrained["fname"])
        else:
            jitted_weights = False

        if jitted_weights:  # check if weights are jitted
            visual = torch.jit.load(Path(weights), map_location="cpu").visual
            patch_device_and_float(visual, device="cpu")
            state_dict = visual.state_dict()

        else:
            state_dict = torch.load(Path(weights), map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            state_dict = remove_criterion_in_state_dict(state_dict)
            state_dict = take_visual_part_of_vit_clip(state_dict, needed_keys=self.visual.state_dict().keys())
            state_dict = remove_prefix_from_state_dict(state_dict, trial_key="conv1.weight")

        self.visual.load_state_dict(state_dict=state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (x.shape[-2] == self.input_size) and (
            x.shape[-1] == self.input_size
        ), f"The model expects input images to be resized to {self.input_size}x{self.input_size}"

        res = self.visual.forward(x)
        if self.normalize:
            res = res / torch.linalg.norm(res, 2, dim=1, keepdim=True).detach()
        return res

    @property
    def feat_dim(self) -> int:
        return self.visual.state_dict()["proj"].shape[-1]


def take_visual_part_of_vit_clip(state_dict: TStateDict, needed_keys: Iterable[str]) -> TStateDict:
    for k in list(state_dict):
        if "visual" in k:
            new_key = k[k.find("visual") + len("visual") + 1 :]
            state_dict[new_key] = state_dict.pop(k)
    state_dict = filter_state_dict(state_dict, needed_keys=needed_keys)
    return state_dict


__all__ = ["ViTCLIPExtractor"]
