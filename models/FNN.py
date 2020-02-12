import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from .BaseForecasters import *

class Linear(Forecaster):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.lin1(x)
        return x
    
    def feature(self, x):
        x = x.view(x.size(0), -1)
        return x

class OneHiddenModel(Forecaster):
    def __init__(self, in_dim, out_dim, n_hidden=100, relu=False, input_scaling=1.0):
        super().__init__()
        self.input_scaling = input_scaling
        if relu:
            self.feature = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, n_hidden),
                nn.ReLU())
        else:
            self.feature = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, n_hidden),
                nn.Sigmoid())

        self.classifier = nn.Linear(n_hidden, out_dim)
        

    def forward(self, x):
        x *= self.input_scaling
        x = self.feature(x)
        x = self.classifier(x)        
        return x

class TwoHiddenModel(Forecaster):
    def __init__(self, in_dim, out_dim, n_hidden=100, relu=False, input_scaling=1.0):
        super().__init__()
        self.input_scaling = input_scaling
        if relu:
            self.feature = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )
        else:
            self.feature = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, n_hidden),
                nn.Sigmoid(),
                nn.Linear(n_hidden, n_hidden),
                nn.Sigmoid()
            )

        self.classifier = nn.Linear(n_hidden, out_dim)
        

    def forward(self, x):
        x *= self.input_scaling
        x = self.feature(x)
        x = self.classifier(x)        
        return x
