import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from .BaseForecasters import *



class AlexNet(Forecaster):
    def __init__(self, pretrained=True, n_labels = 1000, load_type='none'):
        super().__init__()
        model = torchvision.models.alexnet(pretrained=pretrained)
        model = model.eval()
        self.load_type = load_type

        if load_type == 'none':
            self.pred = model
        elif 'feature' in load_type:
            self.pred = model.fc
        elif 'logit' in load_type:
            self.pred = lambda xs: xs
        else:
            raise NotImplementedError
    
    def forward(self, xs):
        return self.pred(xs)

    def eval(self):
        self.training = False
        if 'logit' not in self.load_type:
            self.pred.eval()
        return self
        

class GoogLeNet(Forecaster):
    def __init__(self, pretrained=True, n_labels = 1000, load_type='none'):
        super().__init__()
        model = torchvision.models.googlenet(pretrained=pretrained)
        model = model.eval()
        self.load_type = load_type

        if load_type == 'none':
            self.pred = model
        elif 'feature' in load_type:
            self.pred = model.fc
        elif 'logit' in load_type:
            self.pred = lambda xs: xs
        else:
            raise NotImplementedError
    
    def forward(self, xs):
        return self.pred(xs)

    def eval(self):
        self.training = False
        if 'logit' not in self.load_type:
            self.pred.eval()
        return self

class VGG19(Forecaster):
    def __init__(self, pretrained=True, n_labels = 1000, load_type='none'):
        super().__init__()
        model = torchvision.models.vgg19(pretrained=pretrained)
        model = model.eval()
        self.load_type = load_type

        if load_type == 'none':
            self.pred = model
        elif 'feature' in load_type:
            self.pred = model.fc
        elif 'logit' in load_type:
            self.pred = lambda xs: xs
        else:
            raise NotImplementedError

    def forward(self, xs):
        return self.pred(xs)

    def eval(self):
        self.training = False
        if 'logit' not in self.load_type:
            self.pred.eval()
        return self

