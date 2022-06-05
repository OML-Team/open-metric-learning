from platform import platform

import pytest

from oml.utils.misc import set_global_seed

set_global_seed(42)


@pytest.fixture(scope="session")
def num_workers() -> int:
    """
    Multiprocessing slowly create new processes with `spawn` start method. Only this method is available on Windows
    and MacOS systems. To increase performance we use num_workers > 0 only on pure Linux system.

    We check that `microsoft`, `mac`, etc. are not included into platform, because linux process can be
    launched from docker or WSL on this host platforms.
    """
    os_platform = platform()
    platforms_with_spawn = ['microsoft', 'mac', 'osx']
    if "linux" in os_platform and any(x not in os_platform for x in platforms_with_spawn):
        return 2
    else:
        return 0
