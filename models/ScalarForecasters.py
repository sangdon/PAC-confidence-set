import os, sys
import numpy as np
import math

import torch as tc
import torch.tensor as T
import torch.nn as nn

from .BaseForecasters import Forecaster

class TempForecaster(Forecaster):
    def __init__(self, baseF):
        super().__init__()
        self.baseF = baseF
        self.T = nn.Parameter(T(1.0))
    
    def forward(self, xs):
        return self.baseF(xs) / self.T
    
    def train(self, train_flag=True):
        self.training = True
        self.baseF.eval()
        return self

    def eval(self):
        self.training = False
        self.baseF.eval()
        return self
    
    def train_parameters(self):
        return self.baseF.parameters()
    
    def cal_parameters(self):
        return [self.T]

    
class VarScalingForecaster(Forecaster):
    def __init__(self, baseF):
        super().__init__()
        self.baseF = baseF
        self.T = nn.Parameter(T(1.0))
        
    def forward(self, xs):
        
        yhs, yhs_var = self.baseF(xs)
        mu = yhs
        var = yhs_var * self.T
        return mu, var

    def neg_log_prob(self, mus, vars, ys):

        loss_mah = (ys - mus).pow(2).sum(1, keepdim=True).div(2.0*vars)
        assert(all(loss_mah >= 0))
        loss_const = 0.5 * math.log(2.0 * np.pi)
        loss_logdet = 0.5*vars.log()
        loss = loss_mah + loss_logdet + loss_const

        return loss

    def train(self, train_flag=True):
        self.training = True
        self.baseF.eval()
        return self

    def eval(self):
        self.training = False
        self.baseF.eval()
        return self
    
    def train_parameters(self):
        return self.baseF.parameters()

    def cal_parameters(self):
        return [self.T]
