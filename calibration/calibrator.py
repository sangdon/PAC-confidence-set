import numpy as np
import sys
import os
import pickle
import glob
import time
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch as tc
import torch.tensor as T
from torch import nn, optim

sys.path.append("../")
from classification.learner import SGD
from .utils import *
from classification.utils import *

class BaseCalibrator(SGD):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.param_fn = "model_params_cal"
    
    def validate(self, ld):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        error, _, _ = compute_cls_error([ld], self.model, self.device, loss_fn=loss_fn)
        return error
    
    def test(self, lds, ld_names, model=None):
        if model is None:
            model = self.model
        model.eval()
        
        errors = super().test(lds, ld_names, model=model)
    
        eval_print_ECE(lds, ld_names, model, self.model)
        
        return errors

                
