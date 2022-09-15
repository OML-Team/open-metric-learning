from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from oml.interfaces.models import IExtractor


class LayerNorm(nn.LayerNorm):  # TODO: check if this is not a legacy
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
    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 224,
        layers: int = 12,
        width: int = 768,
        patch_size: int = 32,
        heads: int = 8,
        weights_location: str = "./ViT-B-32.pt",
        jitted_weights: bool = True,  # for jitted weights like from original CLIP repo
        strict_load: bool = True,
    ):
        super().__init__()
        load_path = Path(weights_location)
        assert load_path.is_file(), "There are no weights here!"

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=patch_size,
            layers=layers,
            width=width,
            heads=heads,
            output_dim=embed_dim,
        )

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
        return self.visual.forward(x)

    @property
    def feat_dim(self) -> int:
        return self.visual.state_dict()["proj"].shape[-1]

    @property
    def input_size(self) -> int:
        state_dict = self.visual.state_dict()
        vision_patch_size = state_dict["conv1.weight"].shape[-1]
        grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)
        return vision_patch_size * grid_size
