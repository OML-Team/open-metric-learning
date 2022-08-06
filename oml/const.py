from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple, Union

from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).parent.parent
TMP_PATH = Path(tempfile.gettempdir())

DOTENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "configs"
MODELS_CONFIGS_PATH = PROJECT_ROOT / "configs" / "model"

MOCK_DATASET_PATH = TMP_PATH / "mock_dataset"
CKPT_SAVE_ROOT = TMP_PATH / "torch" / "checkpoints"

MOCK_DATASET_URL = "https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing"

PAD_COLOR = (255, 255, 255)

T_Str2Int_or_Int2Str = Union[Dict[str, int], Dict[int, str]]

TCfg = Union[Dict[str, Any], DictConfig]

# ImageNet Params
TNormParam = Tuple[float, float, float]
MEAN: TNormParam = (0.485, 0.456, 0.406)
STD: TNormParam = (0.229, 0.224, 0.225)
