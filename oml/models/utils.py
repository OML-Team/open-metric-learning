from typing import OrderedDict

import torch.nn

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

        print(f"Prefix {prefix} was removed from the state dict.")

        return state_dict
