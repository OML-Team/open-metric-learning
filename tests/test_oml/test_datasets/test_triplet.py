import torch

from oml.datasets.triplet import tri_collate
from oml.losses.triplet import get_tri_ids_in_plain


def test_collate() -> None:
    # test absence of shuffle after collate function
    def fdata(x: int) -> torch.tensor:
        return x * torch.ones(3, 4, 4)

    # order is: anchor, positive, negative
    items = [
        {"input_tensors": (fdata(1), fdata(2), fdata(3))},
        {"input_tensors": (fdata(11), fdata(22), fdata(33))},
        {"input_tensors": (fdata(111), fdata(222), fdata(333))},
    ]

    batch = tri_collate(items)["input_tensors"]

    ii_a, ii_p, ii_n = get_tri_ids_in_plain(n=len(batch))

    assert (batch[ii_a] % 10 == 1).all()
    assert (batch[ii_p] % 10 == 2).all()
    assert (batch[ii_n] % 10 == 3).all()
