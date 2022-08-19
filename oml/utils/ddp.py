import itertools
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
from torch.distributed import all_gather_object


def merge_list_of_dicts(list_of_dicts: List[Dict[str, Any]], device: torch.device = 'cpu') -> Dict[str, Any]:
    available_types = (list, tuple, torch.Tensor, np.ndarray)

    for dict_sample in list_of_dicts:
        assert all(isinstance(v, available_types) for v in dict_sample.values()), \
            f"Only '{available_types}' are available for merging. " \
            f"Got types '{tuple((k, type(v)) for k, v in list_of_dicts[0].items())}'"
        assert set((k, type(v)) for k, v in list_of_dicts[0].items()) == \
               set((k, type(v)) for k, v in dict_sample.items()),\
               "Dictionaries in list have to have same keys with same types of values"

    output = {}

    for key in list_of_dicts[0].keys():
        if isinstance(list_of_dicts[0][key], (list, tuple)):
            output[key] = list(itertools.chain(*tuple(g[key] for g in list_of_dicts)))
        elif isinstance(list_of_dicts[0][key], torch.Tensor):
            output[key] = torch.cat([g[key].to(device) for g in list_of_dicts], dim=0)
        elif isinstance(list_of_dicts[0][key], np.ndarray):
            output[key] = np.concatenate(tuple(g[key] for g in list_of_dicts), axis=0)
        else:
            raise TypeError(f"Only '{available_types}' are available for merging. "
                            f"Got type '{type(list_of_dicts[0][key])}' for key '{key}'")

    return output


def sync_dicts_ddp(outputs_from_device: Dict[str, Any], world_size: int, device: Union[torch.device, str] = 'cpu') -> Dict[str, Any]:
    """
    Function allows you to combine and merge data stored in dict from all devices. You can place this function in
    your code and all devices upon reaching this function will wait for each other to synchronize and merging dicts.

    NOTE: Function under the hood pickles all object, convert bytes to tensor, then unpickle after syncing.
    With nccl (default) DDP backend intermediate tensors are stored on CUDA.
    """
    if world_size == 1:
        return outputs_from_device
    else:
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        all_gather_object(gathered, outputs_from_device, group=torch.distributed.group.WORLD)

        return merge_list_of_dicts(gathered, device)
