import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from oml.interfaces.models import IExtractor


class ArcFaceResNet50(IExtractor):
    output_size = 2048

    def __init__(self, pretrained: bool = True):
        super(ArcFaceResNet50, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        return pool

    @property
    def feat_dim(self) -> int:
        return self.output_size


class ArcFaceResNet50WithEmbedding(IExtractor):
    def __init__(self, feature_size: int = 2048, embedding_size: int = 128, normalise_features: bool = True):
        super(ArcFaceResNet50WithEmbedding, self).__init__()
        self.base = ArcFaceResNet50()
        self.linear = nn.Linear(feature_size, embedding_size)
        self.normalise_features = normalise_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)

        if self.training and (not self.normalise_features):
            return embedding

        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding

    @property
    def feat_dim(self) -> int:
        return self.linear.out_features
