from datetime import timedelta
from typing import Any, Callable, Tuple

from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing import spawn

import os

from oml.utils.misc import set_global_seed


def fn_ddp_wrapper(rank: int, world_size: int, fn: Callable, *args: Tuple[Any, ...]) -> Any:  # type: ignore
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    set_global_seed(1)
    res = fn(*args)
    destroy_process_group()
    return res


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> Any:  # type: ignore
    if world_size == 0:
        set_global_seed(1)
        return fn(*args)
    # note, 'spawn' automatically passes 'rank' as first argument for 'fn'
    spawn(fn_ddp_wrapper, args=(world_size, fn, *args), nprocs=world_size, join=True)
