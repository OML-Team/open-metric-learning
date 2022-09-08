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
MOCK_DATASET_FILES = "df.csv", "df_with_bboxes.csv", "df_with_category.csv", "df_full.csv"

PAD_COLOR = (255, 255, 255)

TCfg = Union[Dict[str, Any], DictConfig]

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

CROP_KEY = "crop"  # the format is [x1, y1, x2, y2]

# Required dataset format:
LABELS_COLUMN = "label"
PATHS_COLUMN = "path"
SPLIT_COLUMN = "split"
IS_QUERY_COLUMN = "is_query"
IS_GALLERY_COLUMN = "is_gallery"
CATEGORIES_COLUMN = "category"
X1_COLUMN = "x_1"
X2_COLUMN = "x_2"
Y1_COLUMN = "y_1"
Y2_COLUMN = "y_2"

OBLIGATORY_COLUMNS = [LABELS_COLUMN, PATHS_COLUMN, SPLIT_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN]
BBOXES_COLUMNS = [X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN]

# Keys for interactions among our classes (datasets, metrics and so on)
OVERALL_CATEGORIES_KEY = "OVERALL"
INPUT_TENSORS_KEY = "input_tensors"
LABELS_KEY = "labels"
IS_QUERY_KEY = "is_query"
IS_GALLERY_KEY = "is_gallery"
EMBEDDINGS_KEY = "embeddings"
CATEGORIES_KEY = "categories"
PATHS_KEY = "paths"
X1_KEY = "x1"
X2_KEY = "x2"
Y1_KEY = "y1"
Y2_KEY = "y2"
