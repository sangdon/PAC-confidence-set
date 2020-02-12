import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn

sys.path.append("../../../")
from models.BaseForecasters import Forecaster
from calibration.calibrator import BaseCalibrator
from classification.utils import *


##
## forecaster loader
##
def load_forecaster(params):
        
    ## init a dynamics forecaster
    dyn_F = DynamicsForecaster()
    
    ## init a trajectory forecaster
    if params.cs.no_acc:
        traj_F = TrajForecaster_no_acc(dyn_F, params.dataset.n_steps, params.dataset.n_states)
    else:
        traj_F = TrajForecaster(dyn_F, params.dataset.n_steps, params.dataset.n_states)

    return traj_F


##
## forecasters
##
def dist_mah(xs, cs, Ms, sqrt=True):
    diag = True if len(Ms.size()) == 3 else False
    assert(diag)    
    assert(xs.size() == cs.size())
    assert(xs.size() == Ms.size())

    diff = xs - cs
    dist = diff.mul(Ms).mul(diff).sum(1).sum(1)
    if sqrt:
        dist = dist.sqrt()
    return dist
    
class DynamicsForecaster(Forecaster):
    def __init__(self):
        super().__init__()

    def forward(self, xs, var_min=0.0):
        xs, yhs, yhs_var = xs
        yhs_var = tc.max(yhs_var, T(var_min, device=xs.device))
        return yhs, yhs_var

    
class TrajForecaster(Forecaster):
    def __init__(self, baseF, n_steps, n_states):
        super().__init__()
        self.baseF = baseF
        # parameterization
        self.T = nn.Parameter(tc.ones(1))

        
    def acc_cal(self, yhs_var, acc=True):
        ## accumulate
        if acc:
            var_acc = yhs_var.cumsum(1).mul(self.T.unsqueeze(0))
        else:
            var_acc = yhs_var.mul(self.T.unsqueeze(0))        

        return var_acc

    def forward(self, xs):
        yhs, yhs_var = self.baseF(xs)
        assert(yhs.size(2) == yhs_var.size(2))
        mu = yhs
        var_acc = self.acc_cal(yhs_var)
        
        return mu, var_acc

    def neg_log_prob(self, yhs, yhs_var, ys):
        N = yhs.size(1)
        S = yhs.size(2)
        
        loss_mah = 0.5 * dist_mah(ys, yhs, 1/yhs_var, sqrt=False)
        assert(all(loss_mah >= 0))
        loss_const = 0.5 * T(2.0 * np.pi).log() * N * S
        loss_logdet = 0.5 * yhs_var.log().sum(1).sum(1)
        loss = loss_mah + loss_logdet + loss_const

        return loss
            
    def cal_parameters(self):
        return [self.T]

class TrajForecaster_no_acc(TrajForecaster):
    def __init__(self, baseF, n_steps, n_states):
        super().__init__(baseF, n_steps, n_states)

    def forward(self, xs):
        yhs, yhs_var = self.baseF(xs)
        assert(yhs.size(2) == yhs_var.size(2))
        mu = yhs
        var_no_acc = self.acc_cal(yhs_var, acc=False)

        return mu, var_no_acc

