from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from oml.interfaces.models import IExtractor
from oml.models.utils import TStateDict, filter_state_dict, patch_device_and_float
from oml.models.vit.vision_transformer_clip import VisionTransformer
from oml.utils.io import download_checkpoint

SBER_MODELS_URL = "https://huggingface.co/sberbank-ai"
OPENAI_MODELS_URL = "https://openaipublic.azureedge.net/clip/models"


class ViTCLIPExtractor(IExtractor):
    pretrained_models: Dict[str, Any] = {
        # checkpoints pretrained by OpenAI
        "openai_vitb16_224": (
            f"{OPENAI_MODELS_URL}/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
            "44c3d8",
        ),
        "openai_vitb32_224": (
            f"{OPENAI_MODELS_URL}/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "3ba34e",
        ),
        "openai_vitl14_224": (
            f"{OPENAI_MODELS_URL}/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "096db1",
        ),
        "openai_vitl14_336": (
            f"{OPENAI_MODELS_URL}/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
            "b31105",
        ),
        # checkpoints pretrained by SberbankAI
        "sber_vitb16_224": (f"{SBER_MODELS_URL}/ruclip-vit-base-patch16-224/resolve/main/pytorch_model.bin", "7882e0"),
        "sber_vitb32_224": (f"{SBER_MODELS_URL}/ruclip-vit-base-patch32-224/resolve/main/pytorch_model.bin", "e2c4da"),
        "sber_vitl14_224": (f"{SBER_MODELS_URL}/ruclip-vit-large-patch14-224/resolve/main/pytorch_model.bin", "9b4a1c"),
    }
    jitted_weights = {"openai_vitb16_224", "openai_vitb32_224", "openai_vitl14_224", "openai_vitl14_336"}

    def __init__(
        self,
        arch: str,
        normalise_features: bool = True,
        weights: Optional[str] = None,
        strict_load: bool = True,
    ):
        """
        Args:
            weights: Path to weights or special key for pretrained ones or ``None`` for random initialization.
             You can check available pretrained checkpoints in ``ViTCLIPExtractor.pretrained_models``.
            arch: Might be one of ``vitb16_224``, ``vitb32_224``, ``vitl14_224``, ``vitl14_336``.
            normalise_features: Set ``True`` to normalise output features
            strict_load: Whether the weights needed to be loaded strictly. Doesn't work with OpenAI's models.
        """

        super().__init__()

        self.normalize = normalise_features

        cfg = get_vit_config_by_name(arch)
        embed_dim = cfg["embed_dim"]
        image_resolution = cfg["image_resolution"]
        layers = cfg["layers"]
        width = cfg["width"]
        patch_size = cfg["patch_size"]
        heads = cfg["heads"]

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=patch_size,
            layers=layers,
            width=width,
            heads=heads,
            output_dim=embed_dim,
        )

        if weights is None:
            return
        if weights in self.pretrained_models:
            jitted_weights = weights in self.jitted_weights
            weights, md5 = self.pretrained_models[weights]
            weights = download_checkpoint(weights, md5)
        else:
            jitted_weights = False

        if jitted_weights:  # check if weights are jitted
            visual = torch.jit.load(Path(weights), map_location="cpu").visual.eval()
            patch_device_and_float(visual, device="cpu")
            state_dict = visual.state_dict()
        else:
            state_dict = torch.load(Path(weights), map_location="cpu").get("state_dict", state_dict)
            state_dict = filter_vit_clip_state_dict(state_dict, needed_keys=self.visual.state_dict().keys())

        self.visual.load_state_dict(state_dict=state_dict, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return self.visual.forward(x)
        else:
            res = self.visual.forward(x)
            return res / torch.linalg.norm(res, 2, dim=1, keepdim=True).detach()

    @property
    def feat_dim(self) -> int:
        return self.visual.state_dict()["proj"].shape[-1]


def filter_vit_clip_state_dict(state_dict: TStateDict, needed_keys: Iterable[str]) -> TStateDict:
    for k in list(state_dict):
        if k.startswith("visual."):
            state_dict[k.lstrip("visual.")] = state_dict.pop(k)
    state_dict = filter_state_dict(state_dict, needed_keys=needed_keys)
    return state_dict


def get_vit_config_by_name(model_name: str) -> Dict[str, Any]:
    """
    Function which returns configuration of known CLIP models.
    Args:
        model_name: One of ``vitb16_224``, ``vitb32_224``, ``vitl14_224``, ``vitl14_336``.
    """
    CLIP_MODELS = {
        "vitb16_224": {
            "embed_dim": 512,
            "image_resolution": 224,
            "layers": 12,
            "width": 768,
            "patch_size": 16,
            "heads": 12,
        },
        "vitb32_224": {
            "embed_dim": 512,
            "image_resolution": 224,
            "layers": 12,
            "width": 768,
            "patch_size": 32,
            "heads": 12,
        },
        "vitl14_224": {
            "embed_dim": 768,
            "image_resolution": 224,
            "layers": 24,
            "width": 1024,
            "patch_size": 14,
            "heads": 16,
        },
        "vitl14_336": {
            "embed_dim": 768,
            "image_resolution": 224,
            "layers": 24,
            "width": 1024,
            "patch_size": 14,
            "heads": 16,
        },
    }

    assert model_name in CLIP_MODELS, f"Model {model_name} is unknown."
    return CLIP_MODELS[model_name]
