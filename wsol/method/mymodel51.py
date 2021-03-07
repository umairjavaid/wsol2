import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mymodel51']


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def get_attention(self, input, target):
        prob = F.softmax(input, dim=-1)
        prob = prob[range(target.shape[0]), target]
        prob = 1 - prob
        prob = prob ** self.gamma
        return prob

    def get_celoss(self, input, target):
        ce_loss = F.log_softmax(input, dim=1)
        ce_loss = -ce_loss[range(target.shape[0]), target]
        return ce_loss

    def forward(self, input, target):
        attn = self.get_attention(input, target)
        ce_loss = self.get_celoss(input, target)
        loss = self.alpha * ce_loss * attn
        return loss.mean()

floss1 = FocalLoss(alpha=0.25, gamma=2)

class AttnLoss(nn.Module):
    def __init__(self, alpha=1):
        super(AttnLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, input, target):
        loss = torch.log(input)
        loss = loss[range(target.shape[0]), target]
        return loss.mean()

aloss1 = AttnLoss(alpha=0.25)

def get_loss(output_dict, gt_labels, **kwargs):
    floss1(output_dict['logits'], gt_labels.long()) + aloss1(output_dict['attn'], gt_labels.long())
