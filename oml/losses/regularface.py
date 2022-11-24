from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularFace(nn.Module):
    def __init__(self, in_features: int, num_classes: int, criterion: Optional[nn.Module] = None):
        super(RegularFace, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
   
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = self.fc(x)
        
        # for numerical stability
        cos.clamp(-1, 1)
        
        # for eliminate element w_i = w_j
        cos_ind = cos.detach()
        
        cos_ind.scatter_(1, torch.arange(self.num_classes).view(-1, 1).long().to(x.device), -100)
        indices = torch.max(cos_ind, dim=0)[1]
        
        mask = torch.zeros((self.num_classes, self.num_classes)).to(x.device)
        mask.scatter_(1, indices.view(-1, 1).long(), 1)
        
        logit = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.num_classes
        
        return self.criterion(logit, y)