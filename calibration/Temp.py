import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn

sys.path.append("../")
from .calibrator import BaseCalibrator
from classification.utils import *

class TempScaling(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.set_opt_params([self.model.T])    
        
    def validate(self, ld, i):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
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
    

    
