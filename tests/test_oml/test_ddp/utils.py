import inspect
from datetime import timedelta
from typing import Any, Callable, Tuple

from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing import spawn

import os

from oml.utils.misc import set_global_seed


def assert_signature(fn: Callable):
    signature = inspect.signature(fn)

    parameters = list(signature.parameters.keys())

    if len(parameters) < 2 or parameters[0] != 'rank' or parameters[1] != 'world_size':
        raise ValueError(
            f"The function '{fn.__name__}' should have 'rank' and 'world_size' as the first two parameters.")


def ddp_fn_wrapper(rank: int, world_size: int, fn: Callable, *args: Tuple[Any, ...]) -> Any:  # type: ignore
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    set_global_seed(1)
    res = fn(rank, world_size, *args)
    destroy_process_group()
    return res


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> Any:  # type: ignore
    assert_signature(fn)
    if world_size == 0:
        set_global_seed(1)
        return fn(0, world_size, *args)
    # note, 'spawn' automatically passes 'rank' as first argument for 'fn'
    spawn(ddp_fn_wrapper, args=(world_size, fn, *args), nprocs=world_size, join=True)
