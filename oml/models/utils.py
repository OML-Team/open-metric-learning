from typing import Iterable, OrderedDict, Union

import torch
from torch import nn

TStateDict = OrderedDict[str, torch.Tensor]


def find_prefix_in_state_dict(state_dict: TStateDict, trial_key: str) -> str:
    k0 = [k for k in state_dict.keys() if trial_key in k][0]
    prefix = k0[: k0.index(trial_key)]

    assert all(k.startswith(prefix) for k in state_dict.keys())

    return prefix


def remove_prefix_from_state_dict(state_dict: TStateDict, trial_key: str) -> TStateDict:
    prefix = find_prefix_in_state_dict(state_dict, trial_key)

    if prefix == "":
        return state_dict

    else:

        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = state_dict[k]
                del state_dict[k]

        print(f"Prefix <{prefix}> was removed from the state dict.")

        return state_dict


def filter_state_dict(state_dict: TStateDict, needed_keys: Iterable[str]) -> TStateDict:

    for k in list(state_dict):
        if k not in needed_keys:
            del state_dict[k]

    return state_dict


def patch_float(module: nn.Module, float_node: torch.Node) -> None:
    """
    This function is for patching jitted weights with hardcoded ``.to(dtype)`` operation.
    """
    try:
        graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
        graphs = []

    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for graph in graphs:
        for node in graph.findAllNodes("aten::to"):
            inputs = list(node.inputs())
            for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                if inputs[i].node()["value"] == 5:
                    inputs[i].node().copyAttributes(float_node)

    for child in module.children():
        patch_float(child, float_node)


def patch_device(module: nn.Module, device_node: torch.Node) -> None:
    """
    This function is for patching jitted weights with hardcoded ``.to(device)`` operation.
    """
    try:
        graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
        graphs = []

    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)

    for g in graphs:
        for node in g.findAllNodes("prim::Constant"):
            if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                node.copyAttributes(device_node)

    for child in module.children():
        patch_device(child, device_node)


def patch_device_and_float(module: nn.Module, device: Union[str, torch.device] = "cuda") -> None:
    """
    This function is for patching jitted weights with hardcoded ``.to(device)`` and ``.to(dtype)`` operations.
    You may need this if you want to correctly load some jitted model which uses half-precision and(or) which
    device was hardcoded.
    """
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]
    patch_device(module, device_node)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_node = list(float_holder.graph.findNode("aten::to").inputs())[1].node()
        patch_float(module, float_node)


__all__ = [
    "find_prefix_in_state_dict",
    "remove_prefix_from_state_dict",
    "filter_state_dict",
    "patch_float",
    "patch_device",
    "patch_device_and_float",
]
