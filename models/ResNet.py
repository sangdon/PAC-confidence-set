import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from .BaseForecasters import *


class ResNet152(Forecaster):
    def __init__(self, pretrained=True, n_labels = 1000, load_type='none'):
        super().__init__()
        self.load_type = load_type
        model = torchvision.models.resnet152(pretrained=pretrained)

        if self.load_type == 'none':
            self.pred = model
        elif 'logit' in self.load_type:
            self.pred = lambda xs: xs
        else:
            raise NotImplementedError
    
    def forward(self, xs):
        return self.pred(xs)
            

