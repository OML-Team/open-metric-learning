from collections import OrderedDict
from logging import warning
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from oml.interfaces.models import IExtractor
from oml.models.utils import remove_prefix_from_state_dict
from oml.utils.io import download_checkpoint

CLIP_MODELS = {
    "openai_vitb16_224": {
        "weights": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "md5": "44c3d804ecac03d9545ac1a3adbca3a6",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 16,
        "heads": 8,
    },
    "openai_vitb32_224": {
        "weights": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "md5": "3ba34e387b24dfe590eeb1ae6a8a122b",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 32,
        "heads": 8,
    },
    "openai_vitl14_224": {
        "weights": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "md5": "096db1af569b284eb76b3881534822d9",
        "embed_dim": 768,
        "image_resolution": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14,
        "heads": 12,
    },
    "openai_vitl14_336": {
        "weights": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "md5": "b311058cae50cb10fbfa2a44231c9473",
        "embed_dim": 768,
        "image_resolution": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14,
        "heads": 12,
    },
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class ViTCLIPExtractor(IExtractor):
    pretrained_models: Dict[str, Any] = {}  # there are too many pretrained architectures, so we will skip them

    def __init__(
        self,
        arch: Optional[str] = None,
        normalise_features: bool = True,
        embed_dim: int = 512,
        image_resolution: int = 224,
        layers: int = 12,
        width: int = 768,
        patch_size: int = 32,
        heads: int = 8,
        weights: Optional[str] = None,
        strict_load: bool = True,
    ):
        """
        Args:
            weights: Path to weights or ``None`` for randomly initialized model's weights.
             Available pretrained checkpoints are currently matching with possible architectures (``arch`` param).
            arch: Might be one of ``openai_vitb16_224``, ``openai_vitb32_224``, ``openai_vitl14_224``, ``openai_vitl14_336``.
            normalise_features: Set ``True`` to normalise output features
            embed_dim: Embedding dimension.
            image_resolution: Input image resolution.
            layers: Number of layers in ViT.
            width: ViT's width. Default is 3 * 128.
            patch_size: Convolutional encoder patch size.
            heads: Number of heads in MHA.
            strict_load: Whether the weights needed to be loaded strictly. Doesn't work with OpenAI's models.
        """

        super().__init__()

        self.normalize = normalise_features

        if not arch:
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=patch_size,
                layers=layers,
                width=width,
                heads=heads,
                output_dim=embed_dim,
            )
            if weights:
                state_dict = torch.load(Path(weights), map_location="cpu")["state_dict"]
                state_dict = remove_prefix_from_state_dict(state_dict, trial_key="conv1.weight")
                self.visual.load_state_dict(state_dict=state_dict, strict=strict_load)
        else:
            cfg = get_vit_config_by_name(arch)
            embed_dim = cfg["embed_dim"]
            image_resolution = cfg["image_resolution"]
            layers = cfg["layers"]
            width = cfg["width"]
            patch_size = cfg["patch_size"]
            heads = cfg["heads"]
            weights = cfg["weights"]
            self.visual = torch.jit.load(Path(weights), map_location="cpu").visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type):
            if not self.normalize:
                return self.visual.forward(x)
            else:
                res = self.visual.forward(x)
                return res / res.norm(dim=1, keepdim=True)

    @property
    def feat_dim(self) -> int:
        return self.visual.state_dict()["proj"].shape[-1]


def get_vit_config_by_name(model_name: str) -> Dict[str, Any]:
    f"""
    Function which returns configuration of known CLIP models..
    Args:
        model_name: One of ``openai_vitb16_224``, ``openai_vitb32_224``, ``openai_vitl14_224``, ``openai_vitl14_336``.
    """
    assert model_name in CLIP_MODELS, f"Model {model_name} is unknown."

    params = CLIP_MODELS[model_name]
    weights, md5 = params.pop("weights"), params.pop("md5", None)  # type: ignore
    if str(weights).startswith("http"):
        params["weights"] = download_checkpoint(weights, md5)  # type: ignore
    else:
        params["weights"] = weights

    return params
