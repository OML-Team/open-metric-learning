# type: ignore
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import pl_train
from oml.registry.losses import LOSSES_REGISTRY
from oml.registry.transforms import AUGS_REGISTRY

AUGS_REGISTRY["broadface"] = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=Image.LANCZOS),
        transforms.RandomResizedCrop(
            scale=(0.16, 1),
            ratio=(0.75, 1.33),
            size=224,
            interpolation=Image.LANCZOS,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        loss = self.criterion(logit, label)

        return loss


LOSSES_REGISTRY["arcface"] = ArcFace


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return


class LinearEmbedding(nn.Module):
    def __init__(self, base, feature_size=512, embedding_size=128, l2norm_on_train=True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)
        self.l2norm_on_train = l2norm_on_train

    def forward(self, x):
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)

        if self.training and (not self.l2norm_on_train):
            return embedding

        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding


base_model = ResNet50(pretrained=True)

model = LinearEmbedding(
    base_model,
    feature_size=cfg["criterion"]["args"]["in_features"],
    embedding_size=opts.embedding_size,
    l2norm_on_train=False,
)


@hydra.main(config_path="configs", config_name="train_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
