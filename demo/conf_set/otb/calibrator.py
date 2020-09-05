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
## wrapper
##
def cal_forecaster(params, F, ld_train, ld_val, no_cal):

    ## init
    fl = TempScalingReg(params, F)

    ## calibrate
    if not no_cal:
        fl.train(ld_train, ld_val) 

    F.eval()
    return F

def compute_tracking_error(lds, model, device, feature_map=None, loss_fn=None):
    with tc.no_grad():
        model.eval()
        n_error = 0.0
        n_total = 0.0

        for ld in lds:
            for xs, ys in ld:
                if hasattr(xs, "to"):
                    xs = xs.to(device)
                else:
                    assert(hasattr(xs[0], "to"))
                    xs = [x.to(device) for x in xs]
                ys = ys.to(device)
                
                if feature_map is None:
                    zs = xs
                else:
                    zs = feature_map(xs)
                n_total += float(ys.size(0))
                if loss_fn is None:
                    yhs = model.label_pred(zs)
                    n_error += (ys != yhs).float().sum()
                else:
                    assert(feature_map is None)
                    fhs = model(zs)
                    
                    ###tracking specific code: call after forwward pass
                    #ys = model.baseF.encode_bb(zs, ys, model.baseF.opts)

                    n_error += loss_fn(fhs, ys).sum()

        return n_error/n_total, n_error, n_total


##
## calibration for regression
##
class TempScalingReg(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.set_opt_params(self.model.cal_parameters())


    def loss(self, fhs, ys, reduction='mean'):
        yhs, yhs_var = fhs

        #print(self.model.cal_parameters())

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

            ###tracker specific code: call after feedforward
            #ys_src = self.model.baseF.encode_bb(xs_src, ys_src, self.model.baseF.opts)

            loss = loss_fn(fhs, ys_src)
            # backprop
            loss.backward()
            # clip T_grad
            [t.grad.data.clamp_(max=1.0) for t in self.model.cal_parameters()]

            # update parameters
            opt.step()
            ## clip T
            #[t.data.clamp_(min=1e-2, max=1e2) for t in self.model.cal_parameters()]
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
            error, n_error, n_total = compute_tracking_error([ld], model, self.device, loss_fn=loss_fn)
            
            if ld_names is not None:
                print("# %s classification error: %d / %d  = %.2f%%"%(
                    ld_names[i], n_error, n_total, error * 100.0))
            else:
                print("# classification error: %d / %d  = %.2f%%"%(n_error, n_total, error * 100.0))
            errors.append(error.unsqueeze(0))
                    
        return errors

        
    def validate(self, ld, i):
        self.model.eval()
        loss_fn = lambda fhs, ys: self.loss(fhs, ys, reduction='none')
        loss_mean, _, _ = compute_tracking_error([ld], self.model, self.device, loss_fn=loss_fn)
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

