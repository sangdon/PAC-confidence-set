import os, sys
import numpy as np
import math
from scipy.stats import chi2

import torch as tc
import torch.tensor as T
import torch.nn as nn

sys.path.append("../")
from conf_set.utils import *
from models.BaseForecasters import Forecaster
from calibration.calibrator import BaseCalibrator

##
## confidence set
## 
class ConfSet(Forecaster):
    def __init__(self, model_F, eps, delta, n, load_root=None):
        super().__init__()
        self.model_F = model_F
        self.T = nn.Parameter(T(0.0))
        self.eps = nn.Parameter(T(eps), requires_grad=False)
        self.delta = nn.Parameter(T(delta), requires_grad=False)
        self.n = nn.Parameter(T(n), requires_grad=False)

        assert(load_root is None)
        # if load_root is not None:
        #     self.load_cs(load_root)

    def get_T(self, *_):
        return self.T

    def set_T(self, T_val):
        self.T.data = T_val*tc.ones_like(self.T.data)

    def load_cs(self, load_root, exp_name):
        load_fn = os.path.join(load_root, "%s_n_%d_eps_%f_delta_%f/model.pk"%(exp_name, self.n, self.eps, self.delta))
        return self.load(load_fn)
        
    def save_cs(self, save_root, exp_name):
        root = os.path.join(save_root, "%s_n_%d_eps_%f_delta_%f"%(exp_name, self.n, self.eps, self.delta))
        os.makedirs(root, exist_ok=True)
        save_fn = os.path.join(root, "model.pk")
        self.save(save_fn)
        
        
class ConfSetCls(ConfSet):
    def __init__(self, model_F, eps, delta, n, load_root=None):
        """
        T in [0, 1]
        """
        super().__init__(model_F, eps, delta, n, load_root)
        
    ## return a confidene set for classification
    def forward(self, xs, T_exp=None, return_phs=False):
        if T_exp is None:
            T_exp = self.T

        with tc.no_grad():
            phs = self.model_F.prob_pred(xs)
            cs = phs >= T_exp
            
        if return_phs:
            return cs, phs
        else:
            return cs

    ## return a confidence set membership 
    def membership(self, xs, ys):
        with tc.no_grad():
            css = self.forward(xs)
            css_membership = css.gather(1, ys.view(-1, 1)).squeeze(1)
        return css_membership

    ## return the size of a confidence set
    def size(self, xs, T=None, **others):
        css = self.forward(xs, T).sum(1).float()
        return css

class ConfSetCls_cond(ConfSetCls):
    def __init__(self, model_F, eps, delta, n, load_root=None, params=None):
        super().__init__(model_F, eps, delta, n, load_root)
        self.T_min = params.T_min
        self.T_max = params.T_max
        self.T_diff = params.T_diff

    ## return a confidene set for classification
    def forward(self, xs, T=None, return_phs=False):
        assert(T is None)
        
        with tc.no_grad():
            phs = self.model_F.prob_pred(xs)
            # find T conditioned to x
            Ts = self.choose_T(phs, self.T_min, self.T_max, self.T_diff)
            cs = phs >= Ts.unsqueeze(1)
            
        if return_phs:
            return cs, phs
        else:
            return cs

    def choose_T(self, phs, T_min_param, T_max_param, T_diff_param, n_max_iter=10000):
        ## choose T per x
        Ts = []

        ##TODO: vectorize this code
        for ph in phs:
            T_min = T_min_param
            T_max = T_max_param
            T_diff = T_diff_param
            # binary search T such that ph[ph <= T].sum() <= \epsilon
            T_opt = T_min
            #print(ph.max())
            for i in range(n_max_iter):
                # candidate
                T = (T_min + T_max) / 2.0
                # check a condition
                if ph[ph < T].sum() <= self.eps:
                    # update min
                    T_min = T
                    T_opt = T_min
                    #print("T_opt:", T_opt)
                else:                    
                    # update max
                    T_max = T
                # termination
                if abs(T_min - T_max) <= T_diff:
                    break
            Ts.append(T_opt)
        return tc.tensor(Ts, device=phs.device)
    

class ConfSetReg(ConfSet):
    def __init__(self, model_F, eps, delta, n, load_root=None):
        """
        T in [-inf, inf]
        """
        super().__init__(model_F, eps, delta, n, load_root)
        
    def forward(self, xs, T=None, log_scale=False):
        """
        assumption: a covariance matrix is diagonal
        return: an interval for each example
        """
        
        with tc.no_grad():
            ## label distribution prediction
            yhs, yhs_var = self.model_F.forward(xs)

            ## init
            if T is None:
                T = self.get_T(yhs, yhs_var)
        
            ## compute the scaled covariance matrix
            if len(yhs.size()) == 3:
                N, S = yhs.size(1), yhs.size(2)
                const = 2*T - S*N*math.log(2.0*np.pi) - yhs_var.log().sum(1, keepdim=True).sum(2, keepdim=True)
            else:
                assert(len(yhs.size()) == 2)
                S = 1
                N = yhs.size(1)
                const = 2*T - S*N*math.log(2.0*np.pi) - yhs_var.log().sum(1, keepdim=True)
                
            i_zero = (const <= 0).squeeze() # proportional to epsilon, thus small
            vars_scaled = yhs_var.mul(const)
            if log_scale:
                assert(log_scale==False)
            else:
                vars_scaled = vars_scaled.view(vars_scaled.size(0), -1)
                itv_len = vars_scaled.sqrt()
                lb = yhs - itv_len
                ub = yhs + itv_len
                lb[i_zero] = 0.0
                ub[i_zero] = 0.0
                
            return lb, ub

    def membership(self, xs, ys):
        with tc.no_grad():
            yhs, yhs_var = self.model_F.forward(xs)
            neg_log_prob = self.model_F.neg_log_prob(yhs, yhs_var, ys)
            Ts = self.get_T(yhs, yhs_var)
            css_membership = neg_log_prob <= Ts.squeeze()
        return css_membership
 
    def size_fro(self, xs, T=None, log_scale=False, time_summary=[-1]):
        """
        summary the size of confidence set using the frobenious norm of the "shape" of the corresponding ellipsoid

        assumption: a covariance matrix is diagonal
        """
        with tc.no_grad():
            ## trajectory estimates
            yhs, yhs_var = self.model_F.forward(xs)

            ## init
            if T is None:
                T = self.get_T(yhs, yhs_var)
                
            ## compute the scaled covariance matrix
            if len(yhs.size()) == 3:
                N, S = yhs.size(1), yhs.size(2)
                const = 2*T - S*N*math.log(2.0*np.pi) - yhs_var.log().sum(1, keepdim=True).sum(2, keepdim=True)

            else:
                assert(len(yhs.size()) == 2)
                S = 1
                N = yhs.size(1)
                const = 2*T - S*N*math.log(2.0*np.pi) - yhs_var.log().sum(1, keepdim=True)
                
            i_zero = (const <= 0).squeeze() # proportional to epsilon, thus small
            
            vars_scaled = yhs_var.mul(const)
            if log_scale:
                log_vars_scaled = vars_scaled.log()
                if not time_summary:
                    # output entire trajectory without summarizing it
                    log_vars_scaled = log_vars_scaled[~i_zero]
                    size_sq = tc.logsumexp(2.0*log_vars_scaled, 2) # by default, sum along state dimension
                    size = size_sq
                else:
                    # pick time steps specified by time_summary
                    if time_summary[0] is not -1:
                        log_vars_scaled = tc.cat([v[time_summary].unsqueeze(0) for v in log_vars_scaled], 0)
                    log_vars_scaled = log_vars_scaled.view(log_vars_scaled.size(0), -1)
                    size = 0.5 * tc.logsumexp(2.0*log_vars_scaled, 1)
                    size[i_zero] = -np.inf
            else:
                vars_scaled = vars_scaled.view(vars_scaled.size(0), -1)
                size = vars_scaled.pow(2.0).sum(1).sqrt()
                size[i_zero] = 0.0
        return size        

    def size(self, xs, T=None, size_type='fro', log_scale=False, time_summary=[]):
        if size_type == 'fro':
            return self.size_fro(xs, T=T, log_scale=log_scale, time_summary=time_summary)        
        else:
            raise NotImplementedError


class ConfSetReg_cond(ConfSetReg):
    def __init__(self, model_F, eps, delta, n, load_root=None, params=None):
        """
        """
        super().__init__(model_F, eps, delta, n, load_root)
        self.T_min = params.T_min
        self.T_max = params.T_max
        self.T_diff = params.T_diff
        
    def get_T(self, yhs, yhs_var):
        Ts = self.choose_T(yhs, yhs_var, self.T_min, self.T_max, self.T_diff)
        return Ts

    def choose_T(self, yhs, yhs_var, T_min_param, T_max_param, T_diff_param):
        ## choose T per x
        Ts = []
        
        ##TODO: vectorize this code
        for yh, yh_var in zip(yhs, yhs_var):

            T_min = T_min_param
            T_max = T_max_param
            T_diff = T_diff_param
            # binary search T such that ph[ph <= T].sum() <= \epsilon
            T_opt = T_max
            while True:
                # candidate
                T = (T_min + T_max) / 2.0
                # check condition
                var = yh_var.view(-1)
                k = var.size(0)
                const = 2*T - k*math.log(2.0*np.pi) - var.log().sum()
                if const < 0:
                    logcdf = -float("inf")
                else:
                    logcdf = chi2.logcdf(const.cpu(), k)
                
                if logcdf > (1.0 - self.eps).log():
                    # increase level -> decrease T
                    # update max
                    T_max = T
                    T_opt = T
                else:
                    # update min
                    T_min = T
                    
                # termination
                if abs(T_min - T_max) <= T_diff:
                    break
            Ts.append(T_opt)
            
        Ts = tc.tensor(Ts, device=yhs.device)
        if len(yhs.size()) == 2:
            Ts = Ts.view(-1, 1)
        elif len(yhs.size()) == 3:
            Ts = Ts.view(-1, 1, 1)
        else:
            raise NotImplementedError
        return Ts
        
        
##
## cs constructor
##
class ConfSetConstructor(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)

        ## default parameters
        self.model_save_root = os.path.join(
            self.params.snapshot_root, self.params.exp_name)
        if not os.path.exists(self.model_save_root):
            os.makedirs(self.model_save_root)

         
    def train(self, ld_train, ld_val1_train, ld_val1, ld_val2, do_train_cal=True):

        if self.params.cs.load_model:
            if self.model.load_cs(self.model_save_root, params2exp_name(self.params)):
                return True

        if do_train_cal:
            ## train a forecaster
            self.params.train_forecaster(self.params.train, self.model.model_F, ld_train, ld_val1, self.params)
            print()
        
            ## calibration the forecaster
            if self.params.cal.T is None:
                self.params.cal_forecaster(self.params.cal, self.model.model_F, ld_val1_train, ld_val1, self.params.cs.no_cal)
            else:
                # manually set calibration parameter
                self.model.model_F.T.data = tc.tensor(self.params.cal.T).to(self.model.model_F.T.device)
            print()
        
        ## construct a confidence set
        with tc.no_grad():
            eps = self.model.eps.data.item()
            delta = self.model.delta.data.item()
            n = self.model.n.data.item()
            cls = self.params.task == "cls"
            T = self.params.cs.T
            
            print("## construct a confidence set: n = %d, eps = %f, delta = %f"%(n, eps, delta))
            
            ## compute the maximum allowed training error
            if self.params.cs.no_error:
                train_error_allow = 0.0
            else:
                if self.params.cs.no_db:
                    # use the VC bound
                    train_error_allow = compute_tr_error_allow_VC(eps, delta, n, self.device)
                else:
                    # use the direct bound
                    train_error_allow = self.find_maximum_train_error_allow(eps, delta, n)
                if train_error_allow is None:
                    return False

            ## find an optimal threshold
            if not self.params.cs.cond_thres and self.params.cs.T is None: 
                self.find_cs_level(ld_val2, n, train_error_allow, cls, self.params.cs.T_min, self.params.cs.T_max, self.params.cs.T_diff)
            else:
                if T is not None:
                    self.model.set_T(T)

            ## save the last model
            if self.params.cs.save_model and not self.params.cs.cond_thres:
                self.model.save_cs(self.model_save_root, params2exp_name(self.params))
        return True

    def find_maximum_train_error_allow(self, eps, delta, n):
        k_min = 0
        k_max = n
        bnd_min = half_line_bound_upto_k(n, k_min, eps)
        if bnd_min > delta:
            return None
        assert(bnd_min <= delta)
        k = n
        while True:
            # choose new k
            k_prev = k
            k = (T(k_min + k_max).float()/2.0).round().long().item()
        
            # terinate condition
            if k == k_prev:
                break
        
            # check whether the current k satisfies the condition
            bnd = half_line_bound_upto_k(n, k, eps)
            if bnd <= delta:
                k_min = k
            else:
                k_max = k

        # confirm that the solution satisfies the condition
        k_best = k_min
        assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
        error_allow = float(k_best) / float(n)
        return error_allow

    def find_cs_level(self, ld, n, train_error_allow, cls, T_min, T_max, T_diff):

        # binary search
        if cls:
            T_best = T_min
        else:
            T_best = T_max
        while True:
            
            # update
            T_cur = (T_max + T_min)/2.0
            
            # terminate condition
            if abs(T_min-T_max) <= T_diff:
                break
            
            # update lower and max bounds
            self.model.set_T(T_cur)
            error, _, _ = empirical_cs_error(self.model, ld, n, self.device)
            if error <= train_error_allow:
                #T_best_prev = T_best
                T_best = T_cur
                if cls:
                    T_min = T_cur
                else:
                    T_max = T_cur

                print("[best threshold] error = %f, train_error_allow = %f, T = %f"%(error, train_error_allow, T_best))
            else:
                if cls:
                    T_max = T_cur
                else:
                    T_min = T_cur

        # save
        self.model.set_T(T_best)

    def conf_set_size(self, ld, n=None, log_scale=False, time_summary=[-1]):
        return compute_conf_set_size(ld=ld, model=self.model, n=n, log_scale=log_scale, time_summary=time_summary, device=self.device)

    def test(self, lds, lds_name, log_scale=False, name=""):

        cs_sizes = []
        cs_errors = []
        for ld, ld_name in zip(lds, lds_name):
            eps = self.model.eps.data.item()
            delta = self.model.delta.data.item()
            n = self.model.n.data.item()
            T_best = self.model.T.data.item()
            
            ## compute confidence set size
            cs_size = self.conf_set_size(ld, log_scale=log_scale).cpu()

            ## compute confidence set error
            cs_error = empirical_cs_error(self.model, ld, device=self.device)
            
            ## print distribution sumary
            print_cs_stats(name, ld_name, cs_size, cs_error, n, eps, delta, T_best, log_scale=log_scale)

            ## cs_sizes
            cs_sizes.append(cs_size.median())
            cs_errors.append(cs_error)
            
        return cs_sizes, cs_errors
            
