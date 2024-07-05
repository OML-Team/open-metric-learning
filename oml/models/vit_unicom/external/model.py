# type: ignore
# flake8: noqa

import hashlib
import os
import urllib
import warnings
from typing import List

import torch
from tqdm import tqdm

from oml.models.vit_unicom.external.vision_transformer import load_model_and_transform

# ============== CODE FROM UNICOM ==============
# https://github.com/deepglint/unicom/blob/main/unicom/model.py

_MODELS = {
    "ViT-B/32": "https://github.com/deepglint/unicom/releases/download/b32/FP16-ViT-B-32.pt",
    "ViT-B/16": "https://github.com/deepglint/unicom/releases/download/b16/FP16-ViT-B-16.pt",
    "ViT-L/14": "https://github.com/deepglint/unicom/releases/download/l14/FP16-ViT-L-14.pt",
    "ViT-L/14@336px": "https://github.com/deepglint/unicom/releases/download/l14_336px/FP16-ViT-L-14-336px.pt",
}

_SHA256 = {
    "FP16-ViT-B-32.pt": "f9d5696a9b58dbbbefee2d31615ca59084f2895a0fdd2ca4c235e0f9b2793f7a",
    "FP16-ViT-B-16.pt": "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    "FP16-ViT-L-14.pt": "ff3ab62ff782876460099e6e0ee17b73a7c01109de2fffd595f16f4129404bbd",
    "FP16-ViT-L-14-336px.pt": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
}


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def rm_module_from_state_dict(state_dict: dict) -> dict:
    result = {}
    for k, value in state_dict.items():

        if "module." in k:
            k_removed = k.split("module.")[-1]
            result[k_removed] = value
        else:
            result[k] = value
    return result


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L43
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = _SHA256[filename]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L94
def load(name: str, device: str = "cpu", download_root: str = None, using_checkpoint: bool = True):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/unicom"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    with open(model_path, "rb") as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")

    model, transform = load_model_and_transform(name, using_checkpoint=using_checkpoint)
    state_dict_fp32 = {}
    for k, v in state_dict.items():
        state_dict_fp32[k] = v.float()

    model.load_state_dict(state_dict)
    return model, transform
