import itertools
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.distributed import all_gather_object, get_rank, get_world_size


class WarningDDP(UserWarning):
    def __init__(self, function_name: str):
        self.message = (
            f"You use '{function_name}' function, but the default process group has not been initialized. "
            "In this case for compatibility function doesn't modify your data. "
            "Please make sure to call init_process_group or ignore this warning if you want to use this "
            "function without DDP"
        )

    def __str__(self) -> str:
        return repr(self.message)


def merge_list_of_dicts(list_of_dicts: List[Dict[str, Any]], device: torch.device = "cpu") -> Dict[str, Any]:
    available_types = (list, tuple, torch.Tensor, np.ndarray)

    for dict_sample in list_of_dicts:
        assert all(
            isinstance(v, available_types) for v in dict_sample.values()
        ), f"Only '{available_types}' are available for merging"
        assert set((k, type(v)) for k, v in list_of_dicts[0].items()) == set(
            (k, type(v)) for k, v in dict_sample.items()
        ), "Dictionaries in list have to have same keys with same types of values"

    output = {}

    for key, _value_for_type_check in list_of_dicts[0].items():
        if isinstance(_value_for_type_check, (list, tuple)):
            output[key] = type(_value_for_type_check)(itertools.chain(*[g[key] for g in list_of_dicts]))
        elif isinstance(_value_for_type_check, torch.Tensor):
            output[key] = torch.cat([g[key].to(device) for g in list_of_dicts], dim=0)
        elif isinstance(_value_for_type_check, np.ndarray):
            output[key] = np.concatenate(tuple(g[key] for g in list_of_dicts), axis=0)
        else:
            raise TypeError(
                f"Only '{available_types}' are available for merging. "
                f"Got type '{type(list_of_dicts[0][key])}' for key '{key}'"
            )

    return output


def sync_dicts_ddp(
    outputs_from_device: Dict[str, Any], world_size: int, device: Union[torch.device, str] = "cpu"
) -> Dict[str, Any]:
    """
    The function allows you to combine and merge data stored in dictionaries from all devices.
    You can place this function in your code and all the devices upon reaching this function will wait for
    each other to synchronize and merge dictionaries.

    Note:
       The function under the hood pickles all object, converts bytes to tensor, then unpickles them after syncing.
       With `NCCL DDP` backend (the default one) intermediate tensors are stored on CUDA.

    """
    if world_size >= 1 and is_ddp():
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        all_gather_object(gathered, outputs_from_device, group=torch.distributed.group.WORLD)
        return merge_list_of_dicts(gathered, device)
    else:
        if world_size > 1:
            warnings.warn(sync_dicts_ddp.__name__, WarningDDP)
        return outputs_from_device


def is_ddp() -> bool:
    try:
        get_world_size()
        return True
    except RuntimeError:
        return False


def is_main_process() -> bool:
    if is_ddp():
        return get_rank() == 0
    else:
        return True


__all__ = ["is_ddp", "sync_dicts_ddp", "merge_list_of_dicts", "is_main_process"]
