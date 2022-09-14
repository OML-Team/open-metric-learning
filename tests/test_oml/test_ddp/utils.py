from typing import Any, Callable, Tuple

from torch.distributed import init_process_group
from torch.multiprocessing import spawn

from oml.const import TMP_PATH
from oml.utils.misc import set_global_seed


def init_ddp(rank: int, world_size: int) -> None:
    if world_size == 0:
        pass
    else:
        init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{TMP_PATH / 'ddp'}")
    set_global_seed(1)


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> None:  # type: ignore
    if world_size == 0:
        return fn(0, world_size, *args)
    # note, 'spawn' automatically passes 'rank' as first argument for 'fn'
    spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)
