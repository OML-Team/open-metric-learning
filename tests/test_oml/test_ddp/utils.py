import inspect
import socket
from datetime import timedelta
from typing import Any, Callable, Tuple

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing import spawn

from oml.utils.misc import set_global_seed


def assert_signature(fn: Callable) -> None:  # type: ignore
    signature = inspect.signature(fn)

    parameters = list(signature.parameters.keys())

    if len(parameters) < 2 or parameters[0] != "rank" or parameters[1] != "world_size":
        raise ValueError(f"The function '{fn.__name__}' should have 'rank' and 'world_size' as the first two arguments")


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def fn_ddp_wrapper(
    rank: int, port: int, world_size: int, fn: Callable, *args: Tuple[Any, ...]  # type: ignore
) -> Any:  # type: ignore
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )
    set_global_seed(1)
    torch.set_num_threads(1)
    res = fn(rank, world_size, *args)
    destroy_process_group()
    return res


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> Any:  # type: ignore
    assert_signature(fn)
    set_global_seed(1)
    torch.set_num_threads(1)
    if world_size > 1:
        port = get_free_port()
        # note, 'spawn' automatically passes 'rank' to its first argument
        spawn(fn_ddp_wrapper, args=(port, world_size, fn, *args), nprocs=world_size, join=True)
    else:
        set_global_seed(1)
        return fn(0, world_size, *args)
