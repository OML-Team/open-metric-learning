import inspect
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing import spawn

from oml.const import TMP_PATH
from oml.utils.io import calc_hash
from oml.utils.misc import set_global_seed


def assert_signature(fn: Callable) -> None:  # type: ignore
    signature = inspect.signature(fn)

    parameters = list(signature.parameters.keys())

    if len(parameters) < 2 or parameters[0] != "rank" or parameters[1] != "world_size":
        raise ValueError(f"The function '{fn.__name__}' should have 'rank' and 'world_size' as the first two arguments")


def generate_connection_filename(world_size: int, fn: Callable, *args: Tuple[Any, ...]) -> Path:  # type: ignore
    python_info = sys.version_info
    python_str = f"{python_info.major}.{python_info.minor}.{python_info.micro}"
    args_hash = calc_hash(pickle.dumps(args))
    filename = TMP_PATH / f"{fn.__name__}_{world_size}_{python_str}_{args_hash}"
    return filename


def fn_ddp_wrapper(
    rank: int, connection_file: Path, world_size: int, fn: Callable, *args: Tuple[Any, ...]  # type: ignore
) -> Any:  # type: ignore
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{connection_file}",
        timeout=timedelta(seconds=120),
    )
    set_global_seed(1)
    torch.set_num_threads(1)
    res = fn(rank, world_size, *args)
    destroy_process_group()
    return res


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> Any:  # type: ignore
    assert_signature(fn)
    if world_size > 1:
        connection_file = generate_connection_filename(world_size, fn, *args)
        connection_file.unlink(missing_ok=True)
        connection_file.parent.mkdir(exist_ok=True, parents=True)
        # note, 'spawn' automatically passes 'rank' to its first argument
        spawn(fn_ddp_wrapper, args=(connection_file, world_size, fn, *args), nprocs=world_size, join=True)
    else:
        set_global_seed(1)
        return fn(0, world_size, *args)
