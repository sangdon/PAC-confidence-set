import os, sys
import types
import time
import numpy as np
import copy

import torch as tc
import torch.tensor as T

from calibration.utils import *

##
## calibration for classification
##
        
from calibration.Temp import TempScaling as ForecasterLearner

def cal_forecaster(params, F, ld_train, ld_val, no_cal):

    ## init
    fl = ForecasterLearner(params, F)

    ## ECE before cal
    fl.test([ld_val], ["val (before cal)"], model=F)

    ## calibrate
    if not no_cal:
        fl.train(ld_train, ld_val) 

    ## ECE after cal
    fl.test([ld_val], ["val (after cal)"], model=F)
        
    F.eval()
    return F
    
