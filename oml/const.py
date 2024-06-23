import os
import tempfile
from pathlib import Path
from sys import platform
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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


OML_PATH = Path(__file__).parent
PROJECT_ROOT = OML_PATH.parent
CACHE_PATH = get_cache_folder()
TMP_PATH = Path(tempfile.gettempdir())

DOTENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "oml" / "configs"

CKPT_SAVE_ROOT = CACHE_PATH / "torch" / "checkpoints"

LOG_IMAGE_FOLDER = "image_logs"
LOG_TOPK_ROWS_PER_METRIC = 5
LOG_TOPK_IMAGES_PER_ROW = 5
N_GT_SHOW_EMBEDDING_METRICS = 2

STORAGE_URL = "https://oml.daloroserver.com"
STORAGE_CKPTS = STORAGE_URL + "/download/checkpoints"

MOCK_DATASET_PATH = CACHE_PATH / "mock_dataset"
MOCK_DATASET_CSV_NAME = "df.csv"
MOCK_DATASET_URL_GDRIVE = "https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing"
MOCK_DATASET_MD5 = "a23478efd4746d18f937fa6c5758c0ed"

REQUESTS_TIMEOUT = 120.0

TColor = Tuple[int, int, int]
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (120, 120, 120)
BLACK = (0, 0, 0)
PAD_COLOR = (255, 255, 255)

BS_KNN = 5_000

TCfg = Union[Dict[str, Any], DictConfig]

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)

MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
STD_CLIP = (0.26862954, 0.26130258, 0.27577711)

TBBox = Tuple[int, int, int, int]
TBBoxes = Sequence[Optional[TBBox]]

CROP_KEY = "crop"  # the format is [x1, y1, x2, y2]

# Required dataset format:
LABELS_COLUMN = "label"
SPLIT_COLUMN = "split"
IS_QUERY_COLUMN = "is_query"
IS_GALLERY_COLUMN = "is_gallery"
CATEGORIES_COLUMN = "category"
SEQUENCE_COLUMN = "sequence"
# text specific columns:
TEXTS_COLUMN = "text"
# image specific columns:
PATHS_COLUMN = "path"
X1_COLUMN = "x_1"
X2_COLUMN = "x_2"
Y1_COLUMN = "y_1"
Y2_COLUMN = "y_2"

OBLIGATORY_COLUMNS = [LABELS_COLUMN, SPLIT_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN]
BBOXES_COLUMNS = [X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN]

# Keys for interactions among our classes (datasets, metrics and so on)
OVERALL_CATEGORIES_KEY = "OVERALL"
INPUT_TENSORS_KEY = "input_tensors"
LABELS_KEY = "labels"
EMBEDDINGS_KEY = "embeddings"
INDEX_KEY = "idx"

INPUT_TENSORS_KEY_1 = "input_tensors_1"
INPUT_TENSORS_KEY_2 = "input_tensors_2"

IMAGE_EXTENSIONS = ["jpg", "jpeg", "JPG", "JPEG", "png"]

# hydra provides ability to set its behaviour for convinient backwards compatibility
# no matter what the current version is
HYDRA_BEHAVIOUR = "1.1"

# Audio specific constants
MOCK_AUDIO_DATASET_PATH = CACHE_PATH / "mock_audio_dataset"
MOCK_AUDIO_DATASET_URL_GDRIVE = "https://drive.google.com/drive/folders/1aYqXBNnERFRIzxr_6cBVO_v_qz2dU7ta"
MOCK_AUDIO_DATASET_MD5 = "87ba6367ac1231c6be4ccca5e8ace837"

AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".flac"]
FRAME_OFFSET_COLUMN = "frame_offset"
DEFAULT_DURATION = 3.0
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_NUM_CHANNELS = 1
DEFAULT_USE_RANDOM_START = True
DEFAULT_MELSPEC_PARAMS = {"n_fft": 2048, "hop_length": 512, "n_mels": 128}
