from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from oml.ddp.utils import get_world_size_safe, sync_dicts_ddp
from oml.utils.misc_torch import unique_by_ids

TStorage = Dict[str, Union[Tensor, np.ndarray, List[Any]]]


class Accumulator:
    def __init__(self, keys_to_accumulate: Tuple[str, ...]):
        """
        Class for accumulating values of different types, for instance,
        torch.Tensor and numpy.array.

        Args:
            keys_to_accumulate: List or tuple of keys to be collected.
                                 We will take values via these keys calling ``self.update_data()``.
        """
        assert len(keys_to_accumulate) == len(set(keys_to_accumulate)), "All the keys have to be unique!"

        self.keys_to_accumulate = keys_to_accumulate
        self.num_samples = None

        self._collected_samples = 0
        self._storage: TStorage = dict()

        self._indices_key = "__element_indices"  # internal key to keep track of elements order if provided

    def refresh(self, num_samples: int) -> None:
        """
        This method refreshes the state.

        Args:
            num_samples:  The total number of elements you are going to collect (for memory allocation).

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

    def update_data(self, data_dict: Dict[str, Any], indices: Optional[List[int]] = None) -> None:
        """
        Args:
            data_dict: We will accumulate data getting values via ``self.keys_to_accumulate``. All elements
                       of the dictionary have to have the same size.
            indices: Global indices of the elements in your batch of data. If provided, the accumulator
                     will remove accumulated duplicates and return the elements in the sorted order after ``.sync()``.
                     Indices may be useful in DDP (because data is gathered shuffled, additionally you may also get
                     some duplicates due to padding). In the single device regime it's also useful if you accumulate
                     data in shuffled order.

        """
        keys = list(self.keys_to_accumulate)

        if indices is None:
            assert self._indices_key not in self.storage, "We are tracking ids, but they are not currently provided."
        else:
            assert isinstance(indices, List)
            if (self.collected_samples > 0) and (self._indices_key not in self.storage):
                raise RuntimeError("You provided ids, but seems like you had not done it before.")

            keys += [self._indices_key]
            data_dict[self._indices_key] = indices

        bs_values = [len(data_dict[k]) for k in keys]
        bs = bs_values[0]
        assert all(bs == bs_value for bs_value in bs_values), f"Lengths of data are not equal, lengths: {bs_values}"

        for k in keys:
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
        """
        The method drops duplicates and sort elements by indices if they have been provided in ``self.update_data()``.
        In DDP it also gathers data collected on several devices.

        """
        # TODO: add option to broadcast instead of sync to avoid duplicating data
        if not self.is_storage_full():
            raise ValueError(f"Cannot sync. Collected: {self.num_samples}/{self.collected_samples} items.")

        params = {"num_samples": [self.num_samples], "keys_to_accumulate": self.keys_to_accumulate}
        storage = self._storage

        world_size = get_world_size_safe()
        need_rebuilding = False

        if world_size > 1:
            params = sync_dicts_ddp(params, world_size=world_size, device="cpu")
            storage = sync_dicts_ddp(self._storage, world_size=world_size, device="cpu")
            need_rebuilding = True

            assert set(params["keys_to_accumulate"]) == set(
                self.keys_to_accumulate
            ), "Keys of accumulators should be the same on each device"

        if self._indices_key in storage:
            for key, data in storage.items():
                storage[key] = unique_by_ids(storage[self._indices_key], data)[1]  # type: ignore
            indices = storage[self._indices_key]
            need_rebuilding = True
        else:
            indices = None

        if not need_rebuilding:
            # If indices were not provided & it's not DDP we may save time & memory avoiding re-building accumulator
            return self

        synced_accum = Accumulator(tuple(set(params["keys_to_accumulate"])))
        synced_accum.refresh(num_samples=len(storage[list(storage.keys())[0]]))
        synced_accum.update_data(storage, indices=indices)

        return synced_accum


__all__ = ["TStorage", "Accumulator"]
