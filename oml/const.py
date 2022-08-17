import os
import tempfile
from pathlib import Path
from sys import platform
from typing import Any, Dict, Tuple, Union

from omegaconf import DictConfig


def get_cache_folder() -> Path:
    if platform == "linux" or platform == "linux2":
        return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "oml"

    elif platform == "darwin":  # mac os
        return Path.home() / "Library" / "Caches" / "oml"

    elif platform.startswith("win"):
        return Path.home() / ".cache" / "oml"

    else:
        raise ValueError(f"Unexpected platform {platform}.")


PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH = get_cache_folder()
TMP_PATH = Path(tempfile.gettempdir())

DOTENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "oml" / "configs"
MODELS_CONFIGS_PATH = PROJECT_ROOT / "configs" / "model"

MOCK_DATASET_PATH = CACHE_PATH / "mock_dataset"
CKPT_SAVE_ROOT = CACHE_PATH / "torch" / "checkpoints"

MOCK_DATASET_URL = "https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing"

PAD_COLOR = (255, 255, 255)

T_Str2Int_or_Int2Str = Union[Dict[str, int], Dict[int, str]]

TCfg = Union[Dict[str, Any], DictConfig]

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)
