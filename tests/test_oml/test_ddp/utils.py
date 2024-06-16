import atexit
from datetime import timedelta
from typing import Any, Callable, Tuple

from torch.distributed import init_process_group
from torch.multiprocessing import spawn

from oml.const import TMP_PATH
from oml.utils.misc import set_global_seed

# NOTE: tests use the same filename, so it's not safe to run them in parallel
# If you will decide to run them in parallel, keep in mind:
# - main process and workers should know the same filename
# - main workers and workers run in separate Python interpreters and have no shared Python objects
CONN_FILE = TMP_PATH / "ddp"


def init_ddp(rank: int, world_size: int) -> None:
    if world_size == 0:
        pass
    else:
        init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10),
            init_method=f"file://{CONN_FILE}",
        )
    set_global_seed(1)


def run_in_ddp(world_size: int, fn: Callable, args: Tuple[Any, ...] = ()) -> None:  # type: ignore
    CONN_FILE.unlink(missing_ok=True)
    CONN_FILE.parent.mkdir(exist_ok=True, parents=True)
    if world_size == 0:
        return fn(0, world_size, *args)
    # note, 'spawn' automatically passes 'rank' as first argument for 'fn'
    spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)
    atexit.register(lambda: CONN_FILE.unlink(missing_ok=True))