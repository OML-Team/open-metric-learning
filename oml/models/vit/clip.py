from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from oml.interfaces.models import IExtractor
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
        "jitted_weights": True,
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
        "jitted_weights": True,
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
        "jitted_weights": True,
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
        "jitted_weights": True,
    },
    "sber_vitb16_224": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-base-patch16-224/resolve/main/pytorch_model.bin",
        "md5": "7882e07674d78c674e33cb892a68bbfc",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 16,
        "heads": 8,
        "jitted_weights": False,
    },
    "sber_vitb16_384": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-base-patch16-384/resolve/main/pytorch_model.bin",
        "md5": "95e83149d64c81bb7483501e578e8672",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 16,
        "heads": 8,
        "jitted_weights": False,
    },
    "sber_vitb32_224": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-base-patch32-224/resolve/main/pytorch_model.bin",
        "md5": "e2c4dab46a3cfa608bdd762973e90d32",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 32,
        "heads": 8,
        "jitted_weights": False,
    },
    "sber_vitb32_384": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-base-patch32-384/resolve/main/pytorch_model.bin",
        "md5": "e10ae10a6645f9d9ff42cc54d46a0aa2",
        "embed_dim": 512,
        "image_resolution": 224,
        "layers": 12,
        "width": 768,
        "patch_size": 32,
        "heads": 8,
        "jitted_weights": False,
    },
    "sber_vitl14_224": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-large-patch14-224/resolve/main/pytorch_model.bin",
        "md5": "9b4a1cd25d15bad4ffd2ba6e34b8a67c",
        "embed_dim": 768,
        "image_resolution": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14,
        "heads": 12,
        "jitted_weights": False,
    },
    "sber_vitl14_336": {
        "weights": "https://huggingface.co/sberbank-ai/ruclip-vit-large-patch14-336/blob/main/pytorch_model.bin",
        "md5": "3f2d9d1fe41c5b7467b5e9e462dbb371",
        "embed_dim": 768,
        "image_resolution": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14,
        "heads": 12,
        "jitted_weights": False,
    },
}


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
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
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
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
        jitted_weights: bool = True,
        strict_load: bool = True,
    ):
        """
        Args:
            weights: Path to weights or ``None`` for randomly initialized model's weights.
             You can check the available pretrained checkpoints in ``CLIP_MODELS``.
            arch: Might be one of ``openai_vitb16_224``, ``openai_vitb32_224``, ``openai_vitl14_224``, ``openai_vitl14_336``, ``sber_vitb16_224``,
             ``sber_vitb16_384``, ``sber_vitb32_224``, ``sber_vitb32_384``, ``sber_vitl14_224`` or ``sber_vitl14_336``.
            normalise_features: Set ``True`` to normalise output features
            strict_load: Set ``True`` if you want the strict load of the weights from the checkpoint
            jitted_weights: If weights were saved with ``torch.jit.save()`` (as in original CLIP repo), you have to load them differently.
            embed_dim: Embedding dimension.
            image_resolution: Input image resolution.
            layers: Number of layers in ViT.
            width: ViT's width. Default is 3 * 128.
            patch_size: Convolutional encoder patch size.
            heads: Number of heads in MHA.

        """

        super().__init__()

        if arch:
            cfg = get_vit_config_by_name(arch)
            embed_dim = cfg["embed_dim"]
            image_resolution = cfg["image_resolution"]
            layers = cfg["layers"]
            width = cfg["width"]
            patch_size = cfg["patch_size"]
            heads = cfg["heads"]
            weights = cfg["weights"]
            jitted_weights = cfg["jitted_weights"]
            strict_load = True

        self.normalize = normalise_features

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=patch_size,
            layers=layers,
            width=width,
            heads=heads,
            output_dim=embed_dim,
        )
        if not weights:
            return
        load_path = Path(weights)

        _mapper = {
            "n_pre.weight": "ln_pre.weight",
            "n_pre.bias": "ln_pre.bias",
            "n_post.weight": "ln_post.weight",
            "n_post.bias": "ln_post.bias",
        }
        if not jitted_weights:
            state_dict = torch.load(load_path, map_location="cpu")
            sd = {
                _mapper.get(k.lstrip("visual."), k.lstrip("visual.")): v.to(dtype=torch.float32)
                for k, v in state_dict.items()
                if k.startswith("visual.")
            }
            self.visual.load_state_dict(sd, strict=strict_load)
        else:
            model = torch.jit.load(load_path, map_location="cpu")
            sd = {
                _mapper.get(k.lstrip("visual."), k.lstrip("visual.")): v.to(dtype=torch.float32)
                for k, v in model.state_dict().items()
                if k.startswith("visual.")
            }
            self.visual.load_state_dict(sd, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Function which returns configuration of known CLIP models. OpenAI's original CLIP and RuCLIP are suported.
    Args:
        model_name: One of ``openai_vitb16_224``, ``openai_vitb32_224``, ``openai_vitl14_224``, ``openai_vitl14_336``, ``sber_vitb16_224``,
         ``sber_vitb16_384``, ``sber_vitb32_224``, ``sber_vitb32_384``, ``sber_vitl14_224`` or ``sber_vitl14_336``.
    """
    assert model_name in CLIP_MODELS, f"Model {model_name} is unknown."

    params = CLIP_MODELS[model_name]
    weights, md5 = params.pop("weights"), params.pop("md5", None)  # type: ignore
    if str(weights).startswith("http"):
        params["weights"] = download_checkpoint(weights, md5)  # type: ignore
    else:
        params["weights"] = weights

    return params
