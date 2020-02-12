import os, sys
import types
import time
import numpy as np
import itertools
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch as tc
import torch.tensor as T

sys.path.append("../../../")
from calibration.utils import *


def print_ECE(model_S, model_F, ld, ld_name):
    ## measure ECE
    with tc.no_grad():
        ECE_aware = CalibrationError_L1()(
            model_S.label_pred, model_F.tar_prob_pred, [ld])
        ECE_aware_over = CalibrationError_L1()(
            model_S.label_pred, model_F.tar_prob_pred, [ld], True)
        print("[%6s] ECE_aware = %4.2f%%, ECE_aware_over = %4.2f%%"%(
            ld_name, ECE_aware*100.0, ECE_aware_over*100.0))


def plot_stats(dataset_id, cs_car_all, save_root, Ns, deltas, epss, font_size, ylim=(0.0, 1.0)):

    ## plot data
    cs_car_all = cs_car_all.detach().cpu()
    # normalize
    if n_labels is not None:
        cs_car_all[:, -1] = cs_car_all[:, -1] / float(n_labels)
    for (N, delta) in itertools.product(Ns, deltas):
        plot_fn = os.path.join(save_root, "cs_plot_%s_n_%d_delta_%f.png"%(dataset_id, N, delta))
        cs_part_idx = (cs_car_all[:, 0] == N)&(cs_car_all[:, 2] == delta)
        if cs_part_idx.sum().long() == 0:
            continue
        cs_part = cs_car_all[cs_part_idx, :]
        cs_part_plot = []
        cs_part_labels = []
        for i, eps in enumerate(epss):
            cs_part_plot.append(cs_part[cs_part[:, 1] == eps, -1].unsqueeze(1))
            cs_part_labels.append("%.2f"%(eps))
        cs_part_plot = tc.cat(cs_part_plot, 1)
        
        ## box plot
        #plt.rc("text", usetex=True)
        
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()
            plt.boxplot(cs_part_plot.numpy(), whis=np.inf,
                        boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
            plt.gca().set_xticklabels(cs_part_labels)
            plt.xticks(fontsize=int(font_size*0.8))
            plt.yticks(fontsize=int(font_size*0.8))
            plt.grid(True)
            plt.xlabel("epsilon", fontsize=font_size)
            
            if ylim is None:
                plt.yscale('log')
                plt.ylim([1e0, 1e15])
                plt.ylabel("ave. conf. set size (log-scale)", fontsize=font_size)
            else:
                plt.ylim(ylim)
                plt.ylabel("ave. conf. set size", fontsize=font_size)
            
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')
        
def print_cs_stats(name, dataset_id, cs_car, cs_error, n, eps, delta, T_best, log_scale=False):

    if log_scale:
        # convert the log base to natural to 10
        cs_car = cs_car.exp().log10()
        Q1 = cs_car.kthvalue(int(np.round(cs_car.size(0)*0.25)))[0]
        Q2 = cs_car.median()
        Q3 = cs_car.kthvalue(int(np.round(cs_car.size(0)*0.75)))[0]
        print("[%s summary, %s (n = %d), n = %d, eps = %f, delta = %f, T = %e, cs_error = %f] log(min) = %f, log(1st-Q) = %f, log(median) = %f, log(3rd-Q) = %f, log(max) = %f, log(mean) = %f "%(
            name, dataset_id, cs_car.size(0), n, eps, delta, T_best, cs_error[0],
            cs_car.min(), Q1, Q2, Q3, cs_car.max(), cs_car.mean()))
    else:
        Q1 = cs_car.kthvalue(int(np.round(cs_car.size(0)*0.25)))[0]
        Q2 = cs_car.median()
        Q3 = cs_car.kthvalue(int(np.round(cs_car.size(0)*0.75)))[0]
        print("[%s summary, %s (n = %d), n = %d, eps = %f, delta = %f, T = %e, cs_error = %f] min = %f, 1st-Q = %f, median = %f, 3rd-Q = %f, max = %f, mean = %f "%(
            name, dataset_id, cs_car.size(0), n, eps, delta, T_best, cs_error[0],
            cs_car.min(), Q1, Q2, Q3, cs_car.max(), cs_car.mean()))


def params2exp_name(params):
    exp_name = "cs"
    exp_name += "_no_acc" if params.cs.no_acc else ""
    exp_name += "_no_cal" if params.cs.no_cal else ""
    exp_name += "_no_db" if params.cs.no_db else ""
    exp_name += "_no_pac_T_%f"%(params.cs.T) if params.cs.T is not None else ""
    exp_name += "_no_error" if params.cs.no_error else ""
    exp_name += "_cond_thres" if params.cs.cond_thres else ""
    return exp_name


def empirical_cs_error(model, ld, n=None, device=tc.device("cpu")):
    model = model.eval()
    
    n_total = 0.0
    error = []
    for xs, ys in ld:
        if hasattr(xs, "to"):
            xs = xs.to(device)
        else:
            assert(hasattr(xs[0], "to"))
            xs = [x.to(device) for x in xs]
        ys = ys.to(device)
                   
        with tc.no_grad():
            css_membership = model.membership(xs, ys)
            error.append((css_membership==0).float())
            n_total += ys.size(0)
        if n is not None:
            if n_total >= n:
                break
    error = tc.cat(error)
    if n is not None:
        assert(n <= n_total)
        n_total = min(n, n_total)
    n_error = error[0:int(n_total)].sum()
    return n_error/n_total, n_error, n_total

def compute_conf_set_size(ld, model, n=None, log_scale=False, time_summary=[-1], device=tc.device("cpu")):
    css = []
    n_passed = 0
            
    for xs, _ in ld:
        if hasattr(xs, "to"):
            xs = xs.to(device)
            n_batch = xs.size(0)
        else:
            assert(hasattr(xs[0], "to"))
            xs = [x.to(device) for x in xs]
            n_batch = xs[0].size(0)

        with tc.no_grad():
            css_x = model.size(xs, log_scale=log_scale, time_summary=time_summary)
            css.append(css_x)
        n_passed += n_batch
        if n is not None:
            if n_passed >= n:
                break

    css = tc.cat(css)
    if n is not None:
        css = css[0:n]
    return css

    
def log_factorial(n):
    log_f = tc.arange(n, 0, -1).float().log().sum()
    return log_f

def log_n_choose_k(n, k):
    if k == 0:
        return tc.tensor(1)
    else:
        #res = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
        res = tc.arange(n, n-k, -1).float().log().sum() - log_factorial(k)
        return res

def half_line_bound_upto_k(n, k, eps):
    ubs = []
    eps = tc.tensor(eps)
    for i in tc.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        ubs.append(log_ub.exp().unsqueeze(0))
    ubs = tc.cat(ubs)
    ub = ubs.sum()
    return ub

def geb_VC(delta, n, d=1.0):
    n = float(n)
    g = (((T((2*n)/d).log() + 1.0) * d + T(4/delta).log())/n).sqrt()
    return g

def compute_tr_error_allow_VC(eps, delta, n, device):
    g = geb_VC(delta, n)
    
    error_allow = eps - g
    if error_allow < 0.0:
        return None
    else:
        error_allow = tc.tensor(error_allow).to(device)
        return error_allow


def n_VC(eps, delta, k, n_min=1, n_max=1e10):
    n = n_max
    n_opt = n_max
    while True:
        # update n
        n = (n_min + n_max) // 2
        # update range
        if float(k)/float(n) + geb_VC(delta, n) <= eps:
            n_max = n
            n_opt = n
        else:
            n_min = n
        # terminate
        if abs(n_min-n_max) <= 1:
            break
    return n_opt

def n_direct(eps, delta, k, n_min=1, n_max=1e10):
    n = n_max
    n_opt = n_max
    while True:
        # update n
        n = (n_min + n_max) // 2
        # update range
        if half_line_bound_upto_k(n, k, eps) < delta:
            n_max = n
            n_opt = n
        else:
            n_min = n
        # terminate
        if abs(n_min-n_max) <= 1:
            break
    return n_opt
