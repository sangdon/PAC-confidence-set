import os, sys
import types
import time
import numpy as np
import copy

import torch as tc
import torch.tensor as T

sys.path.append("../../")
from conf_set.utils import *
from classification.utils import *
from calibration.calibrator import BaseCalibrator

##
## calibration for regression
##
class TempScalingReg(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.set_opt_params(self.model.cal_parameters())

    def loss(self, fhs, ys, reduction='mean'):
        yhs, yhs_var = fhs
        loss = self.model.neg_log_prob(yhs, yhs_var, ys)

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise NotImplementedError
        
    def train_epoch(self, ld_tr, opt, eps_pert=1e-1):
        loss_fn = self.loss
        for xs_src, ys_src in ld_tr:
            if hasattr(xs_src, "to"):
                xs_src = xs_src.to(self.device)
            else:
                assert(hasattr(xs_src[0], "to"))
                xs_src = [x.to(self.device) for x in xs_src]
            ys_src = ys_src.to(self.device)

            # init for backprop
            opt.zero_grad()
            # compute loss
            fhs = self.model(xs_src)
            loss = loss_fn(fhs, ys_src)
            # backprop
            loss.backward()
            # update parameters
            opt.step()
        print("T:", self.model.cal_parameters())
        return loss

    def test(self, lds, ld_names, model=None):
        if model is None:
            model = self.model
            
        model.eval()
        loss_fn = lambda fhs, ys: self.loss(fhs, ys, reduction='none')
        ## classification error
        if ld_names is not None:
            assert(len(lds) == len(ld_names))
        errors = []
        
        for i, ld in enumerate(lds):
            error, n_error, n_total = compute_cls_error([ld], model, self.device, loss_fn=loss_fn)
            
            if ld_names is not None:
                print("# %s classification error: %d / %d  = %.2f%%"%(
                    ld_names[i], n_error, n_total, error * 100.0))
            else:
                print("# classification error: %d / %d  = %.2f%%"%(n_error, n_total, error * 100.0))
            errors.append(error.unsqueeze(0))
            
        ## combined classification error
        error, n_error, n_total = compute_cls_error(lds, model, self.device, loss_fn=loss_fn)
        print("# Combined classification error: %d / %d  = %.2f%%"%(n_error, n_total, error * 100.0))
        
        ## average classification error
        errors = tc.cat(errors)
        print("# Average classification error = %.2f%%"%(errors.mean() * 100.0))
        
        return errors

        
    def validate(self, ld, i):
        self.model.eval()
        loss_fn = lambda fhs, ys: self.loss(fhs, ys, reduction='none')
        loss_mean, _, _ = compute_cls_error([ld], self.model, self.device, loss_fn=loss_fn)
        if self.loss_best >= loss_mean:
            self.loss_best = loss_mean
            self.epoch_best = i
        return loss_mean
    
    def set_stop_criterion(self):
        self.epoch_best = -np.inf
        self.loss_best = np.inf
    
    def stop_criterion(self, epoch):
        if epoch - self.epoch_best >= self.params.n_epochs*self.params.early_term_cri:
            return True
        else:
            return False

## wrapper
def cal_forecaster(params, F, ld_train, ld_val, no_cal):

    ## init
    fl = TempScalingReg(params, F)

    ## calibrate
    if not no_cal:
        fl.train(ld_train, ld_val) 

    F.eval()
    return F
