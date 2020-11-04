"""
Original repository: https://github.com/xiaomengyc/ACoL
"""

import torch
import torch.nn as nn

from .util import get_attention

__all__ = ['AcolBase']

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

class AcolBase(nn.Module):
    def _acol_logits(self, feature, labels, drop_threshold):
        feat_map_a, logits = self._branch(feature=feature,
                                          classifier=self.classifier_A)
        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention = get_attention(feature=feat_map_a, label=labels)
        erased_feature = _erase_attention(
            feature=feature, attention=attention, drop_threshold=drop_threshold)
        feat_map_b, logit_b = self._branch(feature=erased_feature,
                                           classifier=self.classifier_B)
        return {'logits': logits, 'logit_b': logit_b,
                'feat_map_a': feat_map_a, 'feat_map_b': feat_map_b}

    def _branch(self, feature, classifier):
        feat_map = classifier(feature)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.size(0), -1)
        return feat_map, logits


def _erase_attention(feature, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = feature * mask
    return erased_feature


def get_loss(output_dict, gt_labels, **kwargs):
    return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
           nn.CrossEntropyLoss()(output_dict['logit_b'], gt_labels.long())

def get_focalloss(output_dict, gt_labels, **kwargs):
    return floss1(output_dict['logits'], gt_labels.long()) + \
           floss1(output_dict['logit_b'], gt_labels.long())
