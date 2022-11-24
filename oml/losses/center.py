import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, in_features=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.centers)
        self.detach = True

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature matrix with shape (batch_size, in_features).
            y: ground truth labels with shape (batch_size).
        """
        if self.detach:
            x = x.detach()
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(x.device)
        y = y.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
