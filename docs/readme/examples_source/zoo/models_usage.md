[comment]:zoo-start
```python
from oml.const import CKPT_SAVE_ROOT as CKPT_DIR, MOCK_DATASET_PATH as DATA_DIR
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").eval()
transforms, im_reader = get_transforms_for_pretrained("vits16_dino")

img = im_reader(DATA_DIR / "images" / "circle_1.jpg")  # put path to your image here
img_tensor = transforms(img)
# img_tensor = transforms(image=img)["image"]  # for transforms from Albumentations

features = model(img_tensor.unsqueeze(0))

# Check other available models:
print(list(ViTExtractor.pretrained_models.keys()))

# Load checkpoint saved on a disk:
model_ = ViTExtractor(weights=CKPT_DIR / "vits16_dino.ckpt", arch="vits16", normalise_features=False)
```
[comment]:zoo-end
