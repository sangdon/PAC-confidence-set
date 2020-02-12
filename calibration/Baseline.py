import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn

sys.path.append("../")
from models.BaseForecasters import Forecaster
from .calibrator import BaseCalibrator
    
class BaselineCalibrator(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)

    

    