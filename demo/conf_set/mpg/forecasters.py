import os, sys

##
## forecaster loader
##
def load_forecaster(params):
    
    ## init a base forecaster
    F = MPGRegModel()
    F = F.eval()
    
    ## init a calibrated forecaster
    sys.path.append("../../../")
    from models.ScalarForecasters import VarScalingForecaster as CalForecaster
    F_cal = CalForecaster(F)
    F_cal = F_cal.eval()
    
    return F_cal
    

##
## forecasters
##
import torch as tc
from torch import nn
sys.path.append("../../../")
from models.BaseForecasters import Forecaster
class MPGRegModel(Forecaster):
    def __init__(self, in_dim=7):
        super().__init__()
        
        self.mu_lin1 = nn.Linear(in_dim, 100)
        self.mu_act1 = tc.sigmoid
        self.mu_lin2 = nn.Linear(100, 100)
        self.mu_act2 = tc.sigmoid
        self.mu_lin3 = nn.Linear(100, 1)
        
        self.var_lin1 = nn.Linear(in_dim, 100)
        self.var_act1 = tc.sigmoid
        self.var_lin2 = nn.Linear(100, 100)
        self.var_act2 = tc.sigmoid
        self.var_lin3 = nn.Linear(100, 1)
        
    def forward_mu(self, xs):
        xs = xs*1e-3 # input scaling
        xs = self.mu_lin1(xs)
        xs = self.mu_act1(xs)
        xs = self.mu_lin2(xs)
        xs = self.mu_act2(xs)
        xs = self.mu_lin3(xs)

        return xs
        
    def forward_var(self, xs, var_min=1e-9, var_max=1e9):
        xs = xs*1e-3 # input scaling
        xs = self.var_lin1(xs)
        xs = self.var_act1(xs)
        xs = self.var_lin2(xs)
        xs = self.var_act2(xs)
        xs = self.var_lin3(xs)

        xs = (xs.exp() + 1).log()
        return xs.clamp(var_min, var_max)
        
    def forward(self, xs):
        mu = self.forward_mu(xs)
        var = self.forward_var(xs)
        return mu, var

        
