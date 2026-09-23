from torch import nn
from ..builder import LOSSES

@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE Loss for regression tasks."""
    
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss()
    
    def forward(self, cls_score, labels, **kwargs):
        """Forward function.
        
        Args:
            cls_score (torch.Tensor): Predicted scores (N, 1)
            labels (torch.Tensor): Ground truth labels (N,)
        """
        labels = labels.float().view(-1, 1)
        cls_score = cls_score.view(-1, 1)
        loss = self.criterion(cls_score, labels)
        return loss * self.loss_weight
