# type: ignore
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from oml.datasets.base import BaseDataset
from oml.models.vit.vit import ViTExtractor
from oml.transforms.images.torchvision.transforms import get_normalisation_resize_hypvit
from oml.transforms.images.utils import get_im_reader_for_transforms

dataset_root = Path("/nydl/data/Stanford_Online_Products/")
batch_size = 1024
weights = "vits16_sop"

df = pd.read_csv(dataset_root / "df.csv")

transform = get_normalisation_resize_hypvit(im_size=224, crop_size=224)
im_reader = get_im_reader_for_transforms(transform)

dataset = BaseDataset(df=df, transform=transform, f_imread=im_reader)
model = ViTExtractor(weights, arch="vits16", normalise_features=True).eval().cuda()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)

embeddings = torch.zeros((len(df), model.feat_dim))

with torch.no_grad():
    for i, batch in enumerate(tqdm(train_loader)):
        embs = model(batch["input_tensors"].cuda()).detach().cpu()
        ia = i * batch_size
        ib = min(len(embeddings), (i + 1) * batch_size)
        embeddings[ia:ib, :] = embs

torch.save(embeddings, dataset_root / f"embeddings_{weights}.pkl")
