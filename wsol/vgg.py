"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import spg
from .method import mymodel
from .method import mymodel2
from .method import mymodel45
from .method import MyModel2
from .method.util import normalize_tensor
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms

import copy

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    'acol': {
        '14x14': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M1', 512, 512, 512, 'M2'],
        '28x28': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M2', 512, 512, 512, 'M2'],
    },
    'spg': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    },
    'adl': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 'M', 512, 512, 512, 'A'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 512, 512, 512, 'A'],
    },
    'mymodel': {
        '14x14': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M1', 512, 512, 512, 'M2'],
        '28x28': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M2', 512, 512, 512, 'M2'],
    },
    'mymodel2': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel5': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel6': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel7': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel8': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel15': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel16': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel17': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 'M', 512, 512, 512, 'A'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 512, 512, 512, 'A'],
    },
    'mymodel18': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    'mymodel19': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel20': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel21': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel22': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel23': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel24': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel25': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel26': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel27': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel28': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel29': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel30': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel31': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel32': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel34': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel35': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel36': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
    },
    'mymodel37': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
    },
    'mymodel38': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I'],
    },
    'mymodel39': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel40': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel41': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel42': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel43': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel44': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel45': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel46': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel47': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel48': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel49': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    },
    'mymodel50': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams
        return {'logits': logits}


class VggAcol(AcolBase):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggAcol, self).__init__()

        self.features = features
        self.drop_threshold = kwargs['acol_drop_threshold']

        self.classifier_A = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        feature = self.features(x)
        feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        logits_dict = self._acol_logits(feature=feature, labels=labels,
                                        drop_threshold=self.drop_threshold)

        if return_cam:
            normalized_a = normalize_tensor(
                logits_dict['feat_map_a'].detach().clone())
            normalized_b = normalize_tensor(
                logits_dict['feat_map_b'].detach().clone())
            feature_map = torch.max(normalized_a, normalized_b)
            cams = feature_map[range(batch_size), labels]
            return cams

        return logits_dict


class VggSpg(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggSpg, self).__init__()

        self.features = features
        self.lfs = kwargs['large_feature_map']

        self.SPG_A_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.features(x)
        x = self.SPG_A_1(x)
        if not self.lfs:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.SPG_A_3(x)
        logits_c = self.SPG_C(x)

        feat_map = self.SPG_A_4(x)
        logits = self.avgpool(feat_map)
        logits = logits.flatten(1)

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if return_cam:
            feature_map = feat_map.clone().detach()
            cams = feature_map[range(batch_size), labels]
            return cams

        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


class myModel(nn.Module):
  def __init__(self, features, num_classes=4, **kwargs):
    super(myModel, self).__init__()

    self.features = features
    self.drop_threshold = 0.8
    self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(1024, num_classes, kernel_size=1)
    self.conv6 = nn.Conv2d(1024, num_classes, kernel_size=1)
    self.gap = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x, labels=None, return_cam=False):
    batch_size = x.shape[0]
    x = self.features(x)
    x1 = F.relu(self.conv1(x))
    x1 = F.relu(self.conv3(x1))
    x1 = F.relu(self.conv5(x1))
    #using x as a variable name, instead of x2, to save memory space
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv6(x))
    if self.train and x.requires_grad:
      x.register_hook(lambda grad,feat_maps=x1.detach().clone(),labels=labels: mymodel.custom(grad,feat_maps,labels))
    if return_cam:
      normalized_a = normalize_tensor(x1.detach().clone())
      normalized_b = normalize_tensor(x.detach().clone())
      feature_map = torch.max(normalized_a, normalized_b)
      cams = feature_map[range(batch_size), labels]
      return cams
    x = self.gap(x)
    x1 = self.gap(x1)
    x1 = x1.view(x1.size(0), -1)
    x = x.view(x.size(0), -1)
    return {"logits":x1, "x":x}

class myModel2(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel2, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        if self.train and x.requires_grad:
          random_tensor = torch.rand([], dtype=torch.float32) + 0.25
          random_tensor = random_tensor.floor()
          if(random_tensor.numpy()>0.5):
            x.register_hook(lambda grad,feat_maps=x.detach().clone(),labels=labels: mymodel2.custom(grad,feat_maps,labels))
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #logits = self.fc(pre_logit)

        #if return_cam:
        #    feature_map = x.detach().clone()
        #    cam_weights = self.fc.weight[labels]
        #    cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
        #            feature_map).mean(1, keepdim=False)
        #    return cams
        return {'logits': x}


class myModel3(nn.Module):
  def __init__(self, features=None, num_classes=4, **kwargs):
    super(myModel3, self).__init__()
    self.vgg16 = models.vgg16(pretrained=True)
    self.bn1 = nn.BatchNorm1d(num_features=1000)
    self.linear1 = nn.Linear(1000,500)
    self.linear2 = nn.Linear(500,num_classes)

  def sub_forward(self, x):
    x = self.vgg16(x)
    x = self.bn1(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x
  
  def forward(self, imgs, labels=None, return_cam=False):
    x = self.sub_forward(imgs)
  
    if(return_cam == True):
      with torch.no_grad():
        batch_size, D, H, W = imgs.shape
        activations = self.vgg16.features[:-2](imgs)
        activations = F.relu(activations)
        activations = activations.cpu()
        bactch_size, depth, _, _  = activations.shape
        final_cam_cat = torch.tensor(1)
        for j in range(batch_size):
          final_cam = torch.tensor(1)
          for i in range(depth):
            act = activations[j,i,:,:].unsqueeze_(0).unsqueeze_(0)
            act = act.cuda()     
            #check for Nan values
            if act.max() == act.min():
              continue
            act = F.interpolate(act, (H, W), mode='bilinear', align_corners=False)
            act = normalize_tensor(act)
            mul = imgs[j,:,:,:]*act
            score = self.sub_forward(mul)
            score = F.softmax(score)
            score = score[:,labels[j]]
            score.unsqueeze_(0)
            heatmap = act*score[:,:,None,None]
            if(i == 0):
              final_cam = heatmap
            else:
              final_cam = final_cam + heatmap
          if(j == 0):
            final_cam_cat = final_cam.clone()
          else:
            final_cam_cat = torch.cat((final_cam_cat, final_cam_cat), 0)

        final_cam = F.relu(final_cam)
        final_cam = normalize_tensor(final_cam)
        return final_cam 
    return {'logits': x}


class myModel4(nn.Module):
  def __init__(self, features=None, num_classes=4, **kwargs):
    super(myModel4, self).__init__()
    self.vgg16 = models.vgg16(pretrained=True)
    self.bn1 = nn.BatchNorm1d(num_features=1000)
    self.linear1 = nn.Linear(1000,500)
    self.linear2 = nn.Linear(500,num_classes)
    initialize_weights(self.modules(), init_mode='he')

  def get_masked_imgs(self, imgs, activations):
    b, d, r, c = imgs.shape
    _, A, _, _ = activations.shape
    #targets = targets.unsqueeze_(0)
    #targets = targets.repeat(1,A)
    #targets = targets.reshape(-1)
    imgs = imgs.reshape(-1)
    imgs = imgs.repeat(A)
    activations = activations.permute(1,0,2,3)
    activations = activations.repeat(1,1,d,1)
    activations = activations.reshape(-1)
    mul = activations*imgs
    mul = mul.reshape(-1,d,r,c)
    return mul

  def activation_wise_normalization(self, activations):
    b,f,h,w = activations.shape
    activations = activations.view(-1,h*w)
    max_ = activations.max(dim=1)[0]
    min_ = activations.min(dim=1)[0]
    check = ~max_.eq(min_)
    max_ = max_[check]
    min_ = min_[check]
    activations = activations[check,:]
    sub_ =  max_ - min_
    sub_1 = activations - min_[:,None]
    norm = sub_1 / sub_[:,None]
    norm = norm.view(b,-1,h,w)
    return norm  

  def get_scores(self, imgs, targets):
    b, _, _, _ = imgs.shape
    batch_size = 100
    total_scores = []
    for i in range(0, b, batch_size):
      scores = self.sub_forward(imgs[i:i+batch_size,:,:,:])
      scores = F.softmax(scores, dim=1)
      labels = targets.long()
      scores = scores[:,labels]
      total_scores.append(scores)
    total_scores = torch.cat(total_scores,dim=0)
    total_scores = total_scores.view(-1)
    return total_scores

  def get_cam(self, activations, scores):
    b,f,h,w = activations.shape
    cam = activations*scores[None,:,None,None]
    cam = cam.sum(1, keepdim=True)
    return cam

  def sub_forward(self, x):
    x = self.vgg16(x)
    x = self.bn1(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x
  
  def forward(self, imgs, labels=None, return_cam=False):
    x = self.sub_forward(imgs)
  
    if(return_cam == True):
      with torch.no_grad():
        batch_size, D, H, W = imgs.shape
        y = self.vgg16.features[:-2](imgs)
        y = F.relu(y)
        y = F.relu(y)
        y = F.interpolate(y, (H, W), mode='bilinear', align_corners=False)
        y = self.activation_wise_normalization(y)
        z = self.get_masked_imgs(imgs, y)
        z = self.get_scores(z, labels)
        y = self.get_cam(y,z)
        y = F.relu(y)
        y = normalize_tensor(y)
        return y 
    return  {'logits': x}

class myModel5(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel5, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x = x1 + x2
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel6(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel6, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x = x1 + x2 + x3
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel7(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel7, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel8(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel8, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv14 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv15 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x5 = self.features(x)
        x5 = self.conv14(x5)
        x5 = self.relu(x5)
        x5 = self.conv15(x5)
        x5 = self.relu(x5)

        x = x1 + x2 + x3 + x4 + x5
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel9(nn.Module):
    def __init__(self, features=None, num_classes=4, **kwargs):
        super(myModel9, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.conv6 = nn.Conv2d(10,  num_classes, kernel_size=1) 
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')
  
    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.vgg16(x)
        x = self.bn1(x)
        x = x.view(x.size(0), 10, 10, 10)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)

        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

    
class myModel10(nn.Module):
    def __init__(self, features=None, num_classes=4, **kwargs):
        super(myModel10, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.conv5 = nn.Conv2d(10,  10, kernel_size=1) 
        self.conv6 = nn.Conv2d(10,  num_classes, kernel_size=1) 
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')
  
    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.vgg16(x)
        x = self.bn1(x)
        x = x.view(x.size(0),10,10,10)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)

        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel11(nn.Module):
    def __init__(self, features=None, num_classes=4, **kwargs):
        super(myModel11, self).__init__()
        self.backbone, self.branch_A, self.branch_B = self.get_backbone_and_branches()
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #initialize_weights(self.modules(), init_mode='he')

    def get_backbone_and_branches(self):
        vgg16 = models.vgg16(pretrained=True)
        backbone = copy.deepcopy(vgg16).features[:17]
        branch_A = copy.deepcopy(vgg16).features[17:]
        branch_B = copy.deepcopy(branch_A)
        return backbone, branch_A, branch_B

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.backbone(x)
        
        x1 = self.branch_A(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.branch_B(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x = x1 + x2
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}
    
class myModel12(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel12, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1)
        self.conv6_ = nn.Conv2d(1024,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1)
        self.conv8_ = nn.Conv2d(1024,  1024, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1)
        self.conv10_ = nn.Conv2d(1024,  1024, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv6_(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv8_(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv10_(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x = x1 + x2 + x3
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel15(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel15, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def get_masked_imgs(self, imgs, activations):
      b, d, r, c = imgs.shape
      _, A, _, _ = activations.shape
      imgs = imgs.reshape(-1)
      imgs = imgs.repeat(A)
      activations = activations.permute(1,0,2,3)
      activations = activations.repeat(1,1,d,1)
      activations = activations.reshape(-1)
      mul = activations*imgs
      mul = mul.reshape(-1,d,r,c)
      return mul

    def activation_wise_normalization(self, activations):
      b,f,h,w = activations.shape
      activations = activations.view(-1,h*w)
      max_ = activations.max(dim=1)[0]
      min_ = activations.min(dim=1)[0]
      check = ~max_.eq(min_)
      max_ = max_[check]
      min_ = min_[check]
      activations = activations[check,:]
      sub_ =  max_ - min_
      sub_1 = activations - min_[:,None]
      norm = sub_1 / sub_[:,None]
      norm = norm.view(b,-1,h,w)
      return norm  

    def get_scores(self, imgs, targets):
      b, _, _, _ = imgs.shape
      total_scores = []
      class MyDataloader(torch.utils.data.Dataset):
        def __init__(self, images):
            self.images = images
        def __len__(self):
            return self.images.shape[0]
        def __getitem__(self, idx):
            return self.images[idx, :, :, :]
            
      train_data = MyDataloader(imgs)
      train_loader = torch.utils.data.DataLoader(train_data,
                                                shuffle=False,
                                                num_workers=0,
                                                batch_size=50)
      for batch_images in train_loader:
        scores = self.sub_forward(batch_images)
        scores = F.softmax(scores, dim=1)
        labels = targets.long()
        scores = scores[:,labels]
        total_scores.append(scores)
      total_scores = torch.cat(total_scores,dim=0)
      total_scores = total_scores.view(-1)
      return total_scores
      
    def get_cam(self, activations, scores):
      b,f,h,w = activations.shape
      cam = activations*scores[None,:,None,None]
      cam = cam.sum(1, keepdim=True)
      return cam
    
    
    def sub_forward(self, x):
      x1 = self.features(x)
      x1 = self.conv6(x1)
      x1 = self.relu(x1)
      x1 = self.conv7(x1)
      x1 = self.relu(x1)
      
      x2 = self.features(x)
      x2 = self.conv8(x2)
      x2 = self.relu(x2)
      x2 = self.conv9(x2)
      x2 = self.relu(x2)

      x3 = self.features(x)
      x3 = self.conv10(x3)
      x3 = self.relu(x3)
      x3 = self.conv11(x3)
      x3 = self.relu(x3)

      x = x1 + x2 + x3
      x = self.avgpool(x)
      x = x.view(x.size(0), -1) 
      return x
      
    def forward(self, imgs, labels=None, return_cam=False):
        x = self.sub_forward(imgs)

        if(return_cam == True):
          with torch.no_grad():
            batch_size, D, H, W = imgs.shape
            y = self.features(imgs)
            y = F.relu(y)
            y = F.interpolate(y, (H, W), mode='bilinear', align_corners=False)
            y = self.activation_wise_normalization(y)
            z = self.get_masked_imgs(imgs, y)
            z = self.get_scores(z, labels)
            y = self.get_cam(y,z)
            y = F.relu(y)
            y = normalize_tensor(y)
            y = y.squeeze_(0).detach().clone()
            return y
        
            
        return {'logits': x}

class myModel16(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel16, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.features(x)
        
        x1 = self.conv6(x)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv8(x)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.conv10(x)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x = x1 + x2 + x3
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel17(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel17, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x = x1 + x2 + x3
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel18(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel18, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel19(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel19, self).__init__()
        self.features = features
        self.batch_norm = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.batch_norm(x1)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.batch_norm(x2)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.batch_norm(x3)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.batch_norm(x4)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel20(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel20, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map_x1 = normalize_tensor(x1.detach().clone())
          normalized_feature_map_x2 = normalize_tensor(x2.detach().clone())
          normalized_feature_map_x3 = normalize_tensor(x3.detach().clone())
          normalized_feature_map_x4 = normalize_tensor(x4.detach().clone())
          max_1 = torch.max(normalized_feature_map_x1, normalized_feature_map_x2)
          max_2 = torch.max(normalized_feature_map_x3,normalized_feature_map_x4)
          max_1 = torch.max(max_1,max_2)
          cams = max_1[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel21(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel21, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.mymod2(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.mymod2(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.mymod2(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel22(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel22, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel23(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel23, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        x = x + (x1 + x2 + x3 + x4)
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel24(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel24, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x = x1 - x2
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel25(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel25, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x = x1 - x2 - x3
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel26(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel26, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 - x2 - x3 - x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel27(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel27, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv14 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv15 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x5 = self.features(x)
        x5 = self.conv14(x5)
        x5 = self.relu(x5)
        x5 = self.conv15(x5)
        x5 = self.relu(x5)

        x = x1 - x2 - x3 - x4 - x5
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel28(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel28, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(200)
        self.batch_norm2 = nn.BatchNorm2d(200)
        self.batch_norm3 = nn.BatchNorm2d(200)
        self.batch_norm4 = nn.BatchNorm2d(200)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        x1 = self.batch_norm1(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        x2 = self.batch_norm2(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        x3 = self.batch_norm3(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        x4 = self.batch_norm4(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel29(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel29, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = F.sigmoid(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = F.sigmoid(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = F.sigmoid(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = F.sigmoid(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel30(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel30, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.batch_norm = nn.BatchNorm2d(200)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        x1 = self.batch_norm(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        x2 = self.batch_norm(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        x3 = self.batch_norm(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        x4 = self.batch_norm(x4)

        x = x1 - x2 - x3 - x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel31(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel31, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        x1 = torch.sign(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        x2 = torch.sign(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        x3 = torch.sign(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        x4 = torch.sign(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel32(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel32, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        x1 = torch.sign(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        x2 = torch.sign(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        x3 = torch.sign(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        x4 = torch.sign(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel33(nn.Module):
    def __init__(self, features=None, num_classes=4, **kwargs):
        super(myModel33, self).__init__()
        self.backbone, self.branch_A, self.branch_B, self.branch_C, self.branch_D = self.get_backbone_and_branches()
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def get_backbone_and_branches(self):
        vgg16 = models.vgg16(pretrained=True)
        backbone = copy.deepcopy(vgg16).features[:17]
        block4 = copy.deepcopy(vgg16).features[17:23]
        block5 = copy.deepcopy(vgg16).features[23:]
        branch_A = nn.Sequential(backbone, MyModel2(),block4,MyModel2(),block5,MyModel2())
        branch_B = copy.deepcopy(branch_A)
        branch_C = copy.deepcopy(branch_A)
        branch_D = copy.deepcopy(branch_A)
        return branch_A, branch_B, branch_C, branch_D

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x = self.backbone(x)
        
        x1 = self.branch_A(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.branch_B(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.branch_C(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.branch_D(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = x1 + x2 + x3 + x4
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel34(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel34, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv_add_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.conv_max_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        #x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        x = x + (x1 + x2 + x3 + x4)
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class one_d_tensor(nn.Module):
      def __init__(self, num_classes):
        super(one_d_tensor,self).__init__()
        self.W = torch.nn.Parameter(torch.rand(num_classes,1,1))
        self.W.requires_grad = True

      def forward(self,x):
        mul = torch.mul(x,self.W)
        return mul

class myModel35(nn.Module):   
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel35, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv_add_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.conv_max_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.mymod2 = MyModel2()
        self.add_weight = one_d_tensor(num_classes)
        self.max_weight = one_d_tensor(num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        #x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        x = self.max_weight(x)
        x = x + self.add_weight((x1 + x2 + x3 + x4))
        
        if return_cam:
          normalized_feature_map = normalize_tensor(x.detach().clone())
          cams = normalized_feature_map[range(batch_size), labels]
          return cams
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class vgg16_block4and5(nn.Module):
    def __init__(self):
        super(vgg16_block4and5, self).__init__()
        self.block4_conv1 = nn.Conv2d(256,  512, kernel_size=3, padding=1) 
        self.block4_conv2 = nn.Conv2d(512,  512, kernel_size=3, padding=1) 
        self.block4_conv3 = nn.Conv2d(512,  512, kernel_size=3, padding=1)  
        self.block4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4_attention = MyModel2()

        self.block5_conv1 = nn.Conv2d(512,  512, kernel_size=3, padding=1) 
        self.block5_conv2 = nn.Conv2d(512,  512, kernel_size=3, padding=1) 
        self.block5_conv3 = nn.Conv2d(512,  512, kernel_size=3, padding=1) 
        #self.block5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.block5_attention = MyModel2()
    def forward(self,x):
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))   
        x = F.relu(self.block4_conv3(x))   
        x = self.block4_maxpool(x)
        x = self.block4_attention(x)

        x = F.relu(self.block5_conv1(x))  
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))  
        #x = self.block5_maxpool(x)
        #x = self.block5_attention(x)
        return x

class myModel36(nn.Module):   
    def __init__(self, features, num_classes=4, **kwargs):
        super(myModel36, self).__init__()
        self.features = features
        self.p1 = vgg16_block4and5()
        self.p2 = vgg16_block4and5()
        self.p3 = vgg16_block4and5()
        #self.p4 = vgg16_block4and5()

        #self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        #self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        #self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)

        self.add_weight = one_d_tensor(num_classes)
        self.max_weight = one_d_tensor(num_classes)

        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False, norm_cam = True):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.p1(x1)
        #x1 = self.conv6(x1)
        #x1 = self.relu(x1)
        x1 = self.conv7(x1)
        #x1 = self.relu(x1)

        x2 = self.features(x)
        x2 = self.p2(x2)
        #x2 = self.conv8(x2)
        #x2 = self.relu(x2)
        x2 = self.conv9(x2)
        #x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.p3(x3)
        #x3 = self.conv10(x3)
        #x3 = self.relu(x3)
        x3 = self.conv11(x3)
        #x3 = self.relu(x3)
        
        z = torch.max(x1,x2)
        z = torch.max(z,x3)
        z = self.max_weight(z)
        y = self.add_weight((x1 + x2 + x3))
        x = z + y
        
        if return_cam:
            normalized_feature_map = normalize_tensor(x.detach().clone())
            cams = normalized_feature_map[range(batch_size), labels]
            return cams
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel37(nn.Module):   
    def __init__(self, features, num_classes=4, **kwargs):
        super(myModel37, self).__init__()
        self.features = features
        self.p1 = vgg16_block4and5()
        self.p2 = vgg16_block4and5()
        self.p3 = vgg16_block4and5()
 
        self.conv7 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv9 = nn.Conv2d(512, num_classes, kernel_size=1) 
        self.conv11 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False, norm_cam = True):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.p1(x1)
        x1 = self.conv7(x1)

        x2 = self.features(x)
        x2 = self.p2(x2)
        x2 = self.conv9(x2)

        x3 = self.features(x)
        x3 = self.p3(x3)
        x3 = self.conv11(x3)
        
        x = torch.max(x1 ,x2)
        x = x + torch.max(x2, x3)
        x = x + torch.max(x1, x3)
              
        if return_cam:
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel38(nn.Module):   
    def __init__(self, features, num_classes=4, **kwargs):
        super(myModel38, self).__init__()
        self.features = features
        self.p1 = vgg16_block4and5()
        self.p2 = vgg16_block4and5()
        self.p3 = vgg16_block4and5()
 
        self.conv7 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv9 = nn.Conv2d(512, num_classes, kernel_size=1) 
        self.conv11 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False, norm_cam = True):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.p1(x1)
        x1 = self.conv7(x1)

        x2 = self.features(x)
        x2 = self.p2(x2)
        x2 = self.conv9(x2)

        x3 = self.features(x)
        x3 = self.p3(x3)
        x3 = self.conv11(x3)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x,torch.max(x2, x3))
        x = torch.max(x,torch.max(x1, x3))
        x = x + x1 + x2 + x3
              
        if return_cam:
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}



class myModel39(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel39, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max(x1 ,x2)
        x = x + torch.max(x2, x3)
        x = x + torch.max(x1, x3)
              
        if return_cam:
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel40(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel40, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x,torch.max(x2, x3))
        x = torch.max(x,torch.max(x1, x3))
        x = x + x1 + x2 + x3
              
        if return_cam:
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel41(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel41, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x,torch.max(x2, x3))
        x = torch.max(x,torch.max(x1, x3))
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel42(nn.Module):   
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel42, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv_add_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.conv_max_weight = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        #self.mymod2 = MyModel2()
        self.add_weight = one_d_tensor(num_classes)
        self.max_weight = one_d_tensor(num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        #x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x = torch.max(x1,x2)
        x = torch.max(x,x3)
        x = torch.max(x,x4)
        x = self.max_weight(x)
        x = x + self.add_weight((x1 + x2 + x3 + x4))
        
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        #self.attention = None
    def _select_map(self, add_, max_):
        random_tensor = torch.rand([], dtype=torch.float32) + 0.5
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * add_ + binary_tensor * max_

    def forward(self, input1, input2, input3):
        if not self.training:
            return (input1 + input2 + input3)
        else:
            add = (input1 + input2 + input3)
            max_ = torch.max(input1,input2)
            max_ = torch.max(max_,input3)
            return self._select_map(add,max_)
 
class myModel43(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel43, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.mymod3 = MyModel3()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        
        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        
        x = self.mymod3(x1 ,x2, x3)
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel44(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel44, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.weight1 = one_d_tensor(num_classes)
        self.weight2 = one_d_tensor(num_classes)
        self.weight3 = one_d_tensor(num_classes)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        x1 = self.weight1(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        x2 = self.weight2(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        x3 = self.weight3(x3)
        x3 = self.relu(x3)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x,torch.max(x2, x3))
        x = torch.max(x,torch.max(x1, x3))
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}


class myModel45(nn.Module):
    def pool(self,x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
        
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel45, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x,torch.max(x2, x3))
        x = torch.max(x,torch.max(x1, x3))
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.pool(x)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        return {'logits': x, 'x1':x1, 'x2':x2, 'x3':x3}

class myModel46(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel46, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)
        
        x3 = self.features(x)
        x3 = self.conv8(x3)
        x3 = self.relu(x3)
        x3 = self.conv9(x3)
        x3 = self.relu(x3)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x ,x3)
         
        
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        attn = torch.sigmoid(x)
        attn = self.avgpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        attn = attn.view(attn.size(0), -1)
        return {'logits': x, 'attn': attn}

class myModel47(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel47, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x ,x3)
        x = torch.max(x,x4)
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

class myModel48(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel48, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv14 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv15 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)

        x5 = self.features(x)
        x5 = self.conv14(x5)
        x5 = self.relu(x5)
        x5 = self.conv15(x5)
        x5 = self.relu(x5)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x ,x3)
        x = torch.max(x,x4)
        x = torch.max(x,x5)
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = x + x5.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}

    
class myModel49(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel49, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max(x1 ,x2) + torch.max(x3 ,x4)
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}    
    
class myModel50(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(myModel50, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max((x1 + x2),(x3 + x4))
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}        
    
def mymodel3bweightassign(dict_,dict2):
  dict_['features.0.weight'] = dict2['features.0.weight']
  dict_['features.0.bias'] = dict2['features.0.bias']
  dict_['features.2.weight'] = dict2['features.2.weight']
  dict_['features.2.bias'] = dict2['features.2.bias']
  dict_['features.5.weight'] =dict2['features.5.weight']
  dict_['features.5.bias'] =dict2['features.5.bias']
  dict_['features.7.weight'] =dict2['features.7.weight']
  dict_['features.7.bias'] =dict2['features.7.bias']
  dict_['features.10.weight'] =dict2['features.10.weight']
  dict_['features.10.bias'] =dict2['features.10.bias']
  dict_['features.12.weight'] =dict2['features.12.weight']
  dict_['features.12.bias'] =dict2['features.12.bias']
  dict_['features.14.weight'] =dict2['features.14.weight']
  dict_['features.14.bias'] =dict2['features.14.bias']
  dict_['p1.block4_conv1.weight'] = dict2['features.17.weight']
  dict_['p1.block4_conv1.bias'] =dict2['features.17.bias']
  dict_['p1.block4_conv2.weight'] =dict2['features.19.weight']
  dict_['p1.block4_conv2.bias'] =dict2['features.19.bias']
  dict_['p1.block4_conv3.weight'] =dict2['features.21.weight']
  dict_['p1.block4_conv3.bias'] = dict2['features.21.bias']
  dict_['p1.block5_conv1.weight'] =dict2['features.24.weight']
  dict_['p1.block5_conv1.bias'] =dict2['features.24.bias']
  dict_['p1.block5_conv2.weight'] =dict2['features.26.weight']
  dict_['p1.block5_conv2.bias'] =dict2['features.26.bias']
  dict_['p1.block5_conv3.weight'] =dict2['features.28.weight']
  dict_['p1.block5_conv3.bias'] =dict2['features.28.bias']
  dict_['p2.block4_conv1.weight'] =dict2['features.17.weight']
  dict_['p2.block4_conv1.bias'] = dict2['features.17.bias']
  dict_['p2.block4_conv2.weight'] =dict2['features.19.weight']
  dict_['p2.block4_conv2.bias'] =dict2['features.19.bias']
  dict_['p2.block4_conv3.weight'] =dict2['features.21.weight']
  dict_['p2.block4_conv3.bias'] =dict2['features.21.bias']
  dict_['p2.block5_conv1.weight'] =dict2['features.24.weight']
  dict_['p2.block5_conv1.bias'] =dict2['features.24.bias']
  dict_['p2.block5_conv2.weight'] =dict2['features.26.weight']
  dict_['p2.block5_conv2.bias'] =dict2['features.26.bias']
  dict_['p2.block5_conv3.weight'] =dict2['features.28.weight']
  dict_['p2.block5_conv3.bias'] =dict2['features.28.bias']
  dict_['p3.block4_conv1.weight'] =dict2['features.17.weight']
  dict_['p3.block4_conv1.bias'] = dict2['features.17.bias']
  dict_['p3.block4_conv2.weight'] =dict2['features.19.weight']
  dict_['p3.block4_conv2.bias'] =dict2['features.19.bias']
  dict_['p3.block4_conv3.weight'] =dict2['features.21.weight']
  dict_['p3.block4_conv3.bias'] =dict2['features.21.bias']
  dict_['p3.block5_conv1.weight'] =dict2['features.24.weight']
  dict_['p3.block5_conv1.bias'] =dict2['features.24.bias']
  dict_['p3.block5_conv2.weight'] =dict2['features.26.weight']
  dict_['p3.block5_conv2.bias'] =dict2['features.26.bias']
  dict_['p3.block5_conv3.weight'] =dict2['features.28.weight']
  dict_['p3.block5_conv3.bias'] =dict2['features.28.bias']
    
def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'features.17', 'SPG_A_1.0')
    state_dict = replace_layer(state_dict, 'features.19', 'SPG_A_1.2')
    state_dict = replace_layer(state_dict, 'features.21', 'SPG_A_1.4')
    state_dict = replace_layer(state_dict, 'features.24', 'SPG_A_2.0')
    state_dict = replace_layer(state_dict, 'features.26', 'SPG_A_2.2')
    state_dict = replace_layer(state_dict, 'features.28', 'SPG_A_2.4')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    if(architecture_type in ('mymodel36','mymodel37','mymodel38')):
        mymodel_state_dict = model.state_dict()
        mymodel3bweightassign(mymodel_state_dict, state_dict)
        state_dict = mymodel_state_dict
    else:
        if architecture_type == 'spg':
            state_dict = batch_replace_layer(state_dict)
        state_dict = remove_layer(state_dict, 'classifier.')
        state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [
                ADL(kwargs['adl_drop_rate'], kwargs['adl_drop_threshold'])]
        elif v == 'I':
            layers += [
                MyModel2()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def set_parameter_requires_grad2(model):
  for i,j in enumerate(model.named_parameters()):
    if(i <=16):
      j[1].requires_grad = False
    

def set_parameter_requires_grad(model):
  for name,param in model.named_parameters():
    if('features') in name:
      param.requires_grad = False

def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    if (architecture_type == 'mymodel3'):
        model = myModel3(**kwargs)
        return model
    if (architecture_type == 'mymodel4'):
        model = myModel4(**kwargs)
        return model
    if (architecture_type == 'mymodel9'):
        model = myModel9(**kwargs)
        return model
    if (architecture_type == 'mymodel10'):
        model = myModel10(**kwargs)
        return model
    if (architecture_type == 'mymodel11'):
        model = myModel11(**kwargs)
        return model
    if (architecture_type == 'mymodel33'):
        model = myModel33(**kwargs)
        set_parameter_requires_grad2(model)
        return model
    
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    model = {'cam': VggCam,
             'acol': VggAcol,
             'spg': VggSpg,
             'adl': VggCam,
             'mymodel': myModel,
             'mymodel2': myModel2,
             'mymodel5': myModel5,
             'mymodel6': myModel6,
             'mymodel7': myModel7,
             'mymodel8': myModel8,
             'mymodel15': myModel15,
             'mymodel16': myModel16,
             'mymodel17': myModel17,
             'mymodel18': myModel18,
             'mymodel19': myModel19,
             'mymodel20': myModel20,
             'mymodel21': myModel21,
             'mymodel22': myModel22,
             'mymodel23': myModel23,
             'mymodel24': myModel24,
             'mymodel25': myModel25,
             'mymodel26': myModel26,
             'mymodel27': myModel27,
             'mymodel28': myModel28,
             'mymodel29': myModel29,
             'mymodel30': myModel30,
             'mymodel31': myModel31,
             'mymodel32': myModel32,
             'mymodel34': myModel34,
             'mymodel35': myModel35,
             'mymodel36': myModel36,
             'mymodel37': myModel37,
             'mymodel38': myModel38,
             'mymodel39': myModel39,
             'mymodel40': myModel40,
             'mymodel41': myModel41,
             'mymodel42': myModel42,
             'mymodel43': myModel43,
             'mymodel44': myModel44,
             'mymodel45': myModel45,
             'mymodel46': myModel46,
             'mymodel47': myModel47,
             'mymodel48': myModel48,
             'mymodel49': myModel49,
             'mymodel50': myModel50}[architecture_type](layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
        if(architecture_type in ('mymodel','mymodel2','mymodel3','mymodel4','mymodel5','mymodel6','mymodel7','mymodel8','mymodel9','mymodel10','mymodel15','mymodel16','mymodel17','mymodel18','mymodel19','mymodel20','mymodel21','mymodel22','mymodel23','mymodel24','mymodel25','mymodel26','mymodel27','mymodel28','mymodel29','mymodel30','mymodel31','mymodel32','mymodel34','mymodel35','mymodel39','mymodel40','mymodel41','mymodel42','mymodel43','mymodel44','mymodel45','mymodel46','mymodel47','mymodel48','mymodel49','mymodel50')):
          set_parameter_requires_grad2(model)
        if(architecture_type in ('mymodel36','mymodel37','mymodel38')):
           set_parameter_requires_grad(model)
          
    return model
