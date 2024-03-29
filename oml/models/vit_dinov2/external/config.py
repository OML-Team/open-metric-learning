# References:
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/config.py

import os

import torch

# use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, "scaled_dot_product_attention")
_USE_FUSED_ATTN = int(os.environ.get("USE_FUSED_ATTN", 0))


def use_fused_attn() -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN:
        return False
    return _USE_FUSED_ATTN > 0
