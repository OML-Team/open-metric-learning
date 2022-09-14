from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch.distributed import get_world_size

from oml.ddp.utils import is_ddp, sync_dicts_ddp

TStorage = Dict[str, Union[torch.Tensor, np.ndarray, List[Any]]]


class Accumulator:
    def __init__(self, keys_to_accumulate: Sequence[str]):
        """
        Class for accumulating values of different types, for instance,
        torch.Tensor and numpy.array.

        Args:
            keys_to_accumulate: List or tuple of keys to be collected.
                                 We will take values via these keys calling
                                 >>> self.update_data()
        """
        self.keys_to_accumulate = keys_to_accumulate
        self.num_samples = None

        self._collected_samples = 0
        self._storage: TStorage = dict()

    def refresh(self, num_samples: int) -> None:
        """
        This method refreshes the state.

        Args:
            num_samples:  The total number of elements you are going to collect (for memory allocation)
        """
        assert isinstance(num_samples, int) and num_samples > 0
        self.num_samples = num_samples  # type: ignore
        self._collected_samples = 0
        self._storage = {}

    def _allocate_memory_if_need(self, key: str, batch_value: Any) -> None:
        if self.num_samples is None:
            raise ValueError(
                f"The parameter for memory allocation has not been set up."
                f"Are you sure you've called {self.refresh.__name__}?"
            )

        if key not in self._storage:
            if isinstance(batch_value, torch.Tensor):
                self._storage[key] = torch.empty(
                    (self.num_samples, *batch_value.shape[1:]),
                    dtype=batch_value.dtype,
                    device="cpu",
                    requires_grad=False,
                )
            elif isinstance(batch_value, np.ndarray):
                self._storage[key] = np.empty((self.num_samples, *batch_value.shape[1:]), dtype=batch_value.dtype)
            elif isinstance(batch_value, (list, tuple)):
                self._storage[key] = []
            else:
                raise TypeError(f"Type '{type(batch_value)}' is not available for accumulating")

    def _put_in_storage(self, key: str, batch_value: Any) -> None:
        bs = len(batch_value)

        if isinstance(batch_value, torch.Tensor):
            bv = batch_value.detach()
            self._storage[key][self._collected_samples : self._collected_samples + bs, ...] = bv  # type: ignore
        elif isinstance(batch_value, np.ndarray):
            bv = batch_value
            self._storage[key][self._collected_samples : self._collected_samples + bs, ...] = bv  # type: ignore
        elif isinstance(batch_value, (list, tuple)):
            self._storage[key].extend(list(batch_value))  # type: ignore
        else:
            raise TypeError(f"Type '{type(batch_value)}' is not available for accumulating")

    def update_data(self, data_dict: Dict[str, Any]) -> None:
        """
        Args:
            data_dict: We will accumulate data getting values via
            >>> self.keys_to_accumulate

        """
        bs_values = [len(data_dict[k]) for k in self.keys_to_accumulate]
        bs = bs_values[0]
        assert all(bs == bs_value for bs_value in bs_values), f"Lengths of data are not equal, lengths: {bs_values}"

        for k in self.keys_to_accumulate:
            v = data_dict[k]
            self._allocate_memory_if_need(k, v)
            self._put_in_storage(k, v)
        self._collected_samples += bs

    @property
    def storage(self) -> TStorage:
        return self._storage

    @property
    def collected_samples(self) -> int:
        return self._collected_samples

    def is_storage_full(self) -> bool:
        return self.num_samples == self.collected_samples

    def sync(self) -> "Accumulator":
        # TODO: add option to broadcast instead of sync to avoid duplicating data
        if not self.is_storage_full():
            raise ValueError("Only full storages could be synced")

        if is_ddp():
            world_size = get_world_size()
            if world_size == 1:
                return self
            else:
                params = {"num_samples": [self.num_samples], "keys_to_accumulate": self.keys_to_accumulate}

                gathered_params = sync_dicts_ddp(params, world_size=world_size, device="cpu")
                gathered_storage = sync_dicts_ddp(self._storage, world_size=world_size, device="cpu")

                assert set(gathered_params["keys_to_accumulate"]) == set(
                    self.keys_to_accumulate
                ), "Keys of accumulators should be the same on each device"

                synced_accum = Accumulator(list(set(gathered_params["keys_to_accumulate"])))
                synced_accum.refresh(sum(gathered_params["num_samples"]))
                synced_accum.update_data(gathered_storage)

                return synced_accum
        else:
            return self


__all__ = ["TStorage", "Accumulator"]
