from platform import platform

import pytest

from oml.utils.misc import set_global_seed

set_global_seed(42)


@pytest.fixture(scope="session")
def num_workers() -> int:
    os_platform = platform().lower()
    if "linux" in os_platform and "microsoft" not in os_platform:
        return 2
    else:
        return 0
