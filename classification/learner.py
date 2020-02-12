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
from .utils import *

    
class SGD:
    def __init__(self, params, model):
        ## set default parameters
        if not hasattr(params, "weight_decay_sgd"):
            params.weight_decay_sgd = 0.0
            
        self.params = params
        self.model = model
        if self.params.use_gpu:
            self.device = tc.device("cuda:0")
        else:
            self.device = tc.device("cpu")
            
        self.params.exp_root = os.path.join(self.params.snapshot_root, self.params.exp_name)
        if not os.path.exists(self.params.exp_root):
            os.makedirs(self.params.exp_root)
        self.param_fn = "model_params"
        self.opt_params = None
            
        self.model.to(self.device)
        
    def set_opt_params(self, opt_params):
        self.opt_params = opt_params
        
    def train_post_epoch_processing(self):
        pass
        
    def set_stop_criterion(self):
        pass
    
    def stop_criterion(self, i):
        return False
        
    def train(self, ld_tr, ld_val):
        self.ld_tr = ld_tr
        self.ld_val = ld_val
        
        ## init predictor
        self.model = self.model.to(self.device)
        
        ## load
        if self.params.load_model:
            if self.model.load(
                os.path.join(self.params.exp_root, self.param_fn)):
                return
            
        ## optimizer parameters
        if self.opt_params is None:
            print("self.opt_params is empty")
            return
        else:
            opt_params = self.opt_params

        ## optimizer
        if self.params.optimizer == "Adam":
            opt = optim.Adam(opt_params, lr=self.params.lr)
        elif self.params.optimizer == "AMSGrad":
            opt = optim.Adam(opt_params, lr=self.params.lr, amsgrad=True)
        elif self.params.optimizer == "SGD":
            opt = optim.SGD(opt_params, lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay_sgd)
        else:
            raise NotImplementedError
            
        scheduler = optim.lr_scheduler.StepLR(
            opt, self.params.lr_decay_epoch, self.params.lr_decay_rate)    
        
        ## train
        val_error_best = np.inf
        self.val_error_best = np.inf
        self.set_stop_criterion()
        for i in range(self.params.n_epochs):
            self.current_epoch = i+1
            t_start = time.time()
            self.model.train()
            loss = self.train_epoch(ld_tr, opt)
            # update lr
            scheduler.step()
            
            # print iteration status for every epoch
            if (i+1) % 1 == 0:
                print('[%d/%d, lr=%f, %f sec.] loss: %f'%
                      (i+1, self.params.n_epochs, 
                       opt.param_groups[0]['lr'], time.time()-t_start, 
                       loss))
            
            # eval on given datasets
            if (i+1) % self.params.n_epoch_eval == 0:
                self.test([ld_tr, ld_val], ["train", "val"])
                
            # save best model for every epoch
            if self.params.keep_best:
                val_error = self.validate(ld_val, i)
                if val_error <= val_error_best:
                    val_error_best = val_error
                    print("[model saved] val_eror_best = %4.2f%%"%(val_error_best*100.0))
                    if self.params.save_model:
                        self.model.save(os.path.join(
                            self.params.snapshot_root, 
                            self.params.exp_name, 
                            self.param_fn))
                    self.val_error_best = val_error_best
                print()
            else:
                if self.params.save_model:
                    self.model.save(os.path.join(self.params.snapshot_root, self.params.exp_name, self.param_fn))
                
            # post-epoch process
            self.train_post_epoch_processing()
            # stop
            if self.stop_criterion(i):
                break
        
        self.model.eval()
        if self.params.save_model:
            if self.params.keep_best == False:
                # save the last model
                self.model.save(os.path.join(self.params.snapshot_root, self.params.exp_name, self.param_fn))
            else:
                # load the best model
                self.model.load(os.path.join(self.params.snapshot_root, self.params.exp_name, self.param_fn))
    
    def train_epoch(self, ld_tr, opt):
        if hasattr(self.params, "label_weight"):
            loss_fn = nn.CrossEntropyLoss(weight=self.params.label_weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        for xs_src, ys_src in ld_tr:
            xs_src = xs_src.to(self.device)
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
        return loss
    
    def validate(self, ld, i_epoch):
        self.model.eval()
        error, _, _ = compute_cls_error([ld], self.model, self.device)
        return error
                 
    def test(self, lds, ld_names, model=None):
        if model is None:
            model = self.model
            
        model.eval()
        
        ## classification error
        if ld_names is not None:
            assert(len(lds) == len(ld_names))
        errors = []
        
        for i, ld in enumerate(lds):
            error, n_error, n_total = compute_cls_error([ld], model, self.device)
            
            if ld_names is not None:
                print("# %s classification error: %d / %d  = %.2f%%"%(
                    ld_names[i], n_error, n_total, error * 100.0))
            else:
                print("# classification error: %d / %d  = %.2f%%"%(n_error, n_total, error * 100.0))
            errors.append(error.unsqueeze(0))
        errors = tc.cat(errors)

        if len(lds) > 1:
            ## combined classification error
            error, n_error, n_total = compute_cls_error(lds, model, self.device)
            print("# Combined classification error: %d / %d  = %.2f%%"%(n_error, n_total, error * 100.0))
        
            ## average classification error
            print("# Average classification error = %.2f%%"%(errors.mean() * 100.0))
        
        return errors
    
