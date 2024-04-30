# from oml.models.meta.projection import ExtractorWithMLP
from oml.models.meta.siamese import (
    ConcatSiamese,
    LinearTrivialDistanceSiamese,
    TrivialDistanceSiamese,
)
from oml.models.resnet.extractor import ResnetExtractor
# from oml.models.resnet.pooling import GEM
from oml.models.vit_clip.external.model import VisionTransformer
from oml.models.vit_clip.extractor import ViTCLIPExtractor
from oml.models.vit_dino.extractor import ViTExtractor
# from oml.models.vit_dino.external_v2.config import use_fused_attn
from oml.models.vit_unicom.extractor import ViTUnicomExtractor
# from oml.models.vit_unicom.external import vision_transformer
# from oml.models.vit_unicom.external.model import load  # type: ignore
from oml.models.vit_unicom.external.vision_transformer import load_model_and_transform
# from oml.models.utils import (
#     remove_criterion_in_state_dict,
#     remove_prefix_from_state_dict,
# )
# from oml.models.vit_dino.external.hubconf import (  # type: ignore
#    dino_vitb8,
#     dino_vitb16,
#     dino_vits8,
#     dino_vits16,
# )
# from oml.models.vit_dino.external_v2.hubconf import (  # type: ignore
#     dinov2_vitb14,
#     dinov2_vitb14_reg,
#     dinov2_vitl14,
#     dinov2_vitl14_reg,
#     dinov2_vits14,
#     dinov2_vits14_reg,
# )
