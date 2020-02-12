import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()

    def prob_pred(self, xs):
        fhs = self(xs)
        if fhs.size(1) == 1:
            phs = self.sigmoid(fhs)
        else:
            phs = F.softmax(fhs, 1)
        return phs
    
    def label_pred(self, xs):
        yhs = self(xs).argmax(1)
        return yhs
    
    def tar_prob_pred(self, xs, yhs):
        phs = self.prob_pred(xs)
        phs_tar = phs.gather(1, yhs.view(-1, 1)).squeeze(1)
        return phs_tar
    
    def cal(self):
        self.eval()
        return self

    
    def load(self, model_full_name):
        if os.path.exists(model_full_name):
            self.load_state_dict(tc.load(model_full_name), strict=True)
            return True
        else:
            return False
        
    def save(self, model_full_name):
        tc.save(self.state_dict(), model_full_name)
        
        
class DummyForecaster(Forecaster):
    def __init__(self):
        super().__init__()
        
    def feature(self, xs):
        return xs
    
    def forward(self, xs):
        return self.feature(xs)
    
class NaiveForecaster(Forecaster):
    def __init__(self, baseF):
        super().__init__()
        self.baseF = baseF
    
    def forward(self, xs):
        return self.baseF(xs)
