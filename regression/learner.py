import os, sys
import numpy as np
import math

import torch as tc

sys.path.append("../")
from classification import learner
from regression.utils import *

class Regressor(learner.SGD):
    """
    assume Gaussian with a diagonal covariance matrix
    """
    def __init__(self, params, model):
        super().__init__(params, model)
        
    def loss(self, mus, vars, ys, logvar=False):
        """
        compute negative log loss
        """
        if logvar:
            logvar = vars
            assert(len(mus.size()) == len(logvars.size()) == len(ys.size()))
            assert(all(~tc.isinf(logvars.exp())))

            sim_term = (ys - mus).pow(2).sum(1, keepdim=True).div(2.0*logvars.exp())
            reg_term = 0.5*logvars + 0.5*math.log(2*np.pi)
            loss = (sim_term + reg_term).mean()
        else:
            assert(len(mus.size()) == len(vars.size()) == len(ys.size()))
            sim_term = (ys - mus).pow(2).sum(1, keepdim=True).div(2.0*vars)
            reg_term = 0.5*vars.log() + 0.5*math.log(2*np.pi)
            loss = (sim_term + reg_term).mean()

        return loss
        
    def train_epoch(self, ld_tr, opt):
        loss_fn = self.loss
        for xs, ys in ld_tr:
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            # init for backprop
            opt.zero_grad()
            # compute loss
            mus, vars = self.model(xs)
            loss = loss_fn(mus, vars, ys)
            # backprop
            loss.backward()
            # update parameters
            opt.step()
        return loss
    
    def validate(self, ld, i):
        self.model.eval()
        error, _, _ = compute_reg_loss(ld, self.model, self.loss, self.device)
        return error
    
    def test(self, lds, ld_names, model=None):
        if model is None:
            model = self.model
        model.eval()
        
        if ld_names is not None:
            assert(len(lds) == len(ld_names))
            
        ## regression loss
        errors = []        
        for i, ld in enumerate(lds):
            error, n_error, n_total = compute_reg_loss(ld, model, self.loss, self.device)
            
            if ld_names is not None:
                print("# %s regression loss: %.6f"%(ld_names[i], error))
            else:
                print("# regression loss: %.6f"%(error))
            errors.append(error.unsqueeze(0))
            
        return errors
