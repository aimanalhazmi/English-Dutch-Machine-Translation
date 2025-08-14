import torch
from torch import nn
from torch.nn import functional as F

class WeightedFocalLoss(nn.Module):

    def __init__(self, class_weights, alpha, gamma, ignore_idx):

        super().__init__() 
        self.register_buffer("class_weights", class_weights)
        self.alpha = alpha 
        self.gamma = gamma 
        self.ignore_idx = ignore_idx 

    def forward(self, predictions, targets):

        # mask padding idx
        mask = (targets != self.ignore_idx)

        predictions_masked = predictions[mask]
        targets_masked = targets[mask]

        # Standard cross entropy (with class weights)
        ce_loss = F.cross_entropy(predictions_masked, targets_masked, weight=self.class_weights, reduction='none')

        log_pt = F.log_softmax(predictions_masked, dim=-1)
        pt = log_pt.gather(1, targets_masked.unsqueeze(1)).squeeze(1)
        pt = torch.exp(pt)

        # calculate loss
        focal_weight = self.alpha * (1-pt) ** self.gamma
        focal_loss = focal_weight * ce_loss 
        focal_loss = focal_loss.mean()

        return focal_loss

