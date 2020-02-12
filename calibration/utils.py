import numpy as np
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import glob
import time

import torch as tc
from torch import nn, optim
import torch.tensor as T
import torch.nn.functional as F


class CalibrationError_L1:
    def __init__(self, n_bins=15, device=tc.device("cuda:0")):
        self.n_bins = n_bins
        self.device = device
        bin_boundaries = tc.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    
    def get_acc_conf_mat(self, yhs, phs, ys):
        accs = yhs.eq(ys)
        confs = phs
        
        acc_conf_mat = tc.zeros(self.n_bins, 3)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            if i == 0:
                in_bin = (confs >= bin_lower.item()) & (confs <= bin_upper.item())
            else:
                in_bin = (confs > bin_lower.item()) & (confs <= bin_upper.item())
            # the number of examples in this bin
            acc_conf_mat[i, 0] = in_bin.float().sum()
            if acc_conf_mat[i, 0] > 0:
                # accumulate correct label predictions
                acc_conf_mat[i, 1] = accs[in_bin].float().sum()
                # accumulate confidence predictions
                acc_conf_mat[i, 2] = confs[in_bin].float().sum()
        
        return acc_conf_mat

    # ECE: emprical calibration error
    def ECEmat2ECE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # normalize the correct label predictions
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # normalize the confidence predictions
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        ECE_mat[:, 0] = ECE_mat[:, 0].div(ECE_mat[:, 0].sum())
        ECE = ECE_mat[:, 0].mul((ECE_mat[:, 1] - ECE_mat[:, 2]).abs()).sum()
        return ECE

    # ECE_oneside: emprical calibration error oneside
    def ECEmat2ECE_overconfidence(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # normalize the correct label predictions
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # normalize the confidence predictions
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        ECE_mat[:, 0] = ECE_mat[:, 0].div(ECE_mat[:, 0].sum())
        ECE = ECE_mat[:, 0].mul((ECE_mat[:, 2] - ECE_mat[:, 1]).clamp(0.0, np.inf)).sum()
        return ECE

    # MOCE: maximum-overconfident calibration error
    def ECEmat2MOCE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # mean accuracy
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # mean confidence
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        MOCE = (ECE_mat[:, 2] - ECE_mat[:, 1]).clamp(0.0, np.inf).max()
        return MOCE

    # EUCE: expected-underconfident calibration error
    def ECEmat2EUCE(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        # mean accuracy
        ECE_mat[ind, 1] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        # mean confidence
        ECE_mat[ind, 2] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        # frequency of each bin
        ECE_mat[:, 0] = ECE_mat[:, 0].div(ECE_mat[:, 0].sum())
        #FIXME: does not count all samples, loose information, need to use with MOCE
        EUCE = ECE_mat[:, 0].mul((ECE_mat[:, 1] - ECE_mat[:, 2]).clamp(0.0, np.inf)).sum()
        return EUCE

    def ECEmat2MOEUCE(self, ECE_mat):
        MOCE = ECEmat2MOCE(ECE_mat)
        EUCE = ECEmat2EUCE(ECE_mat)
        return MOCE + EUCE

    def decomposeECEmat(self, ECE_mat):
        ECE_mat = ECE_mat.clone()
        ind = ECE_mat[:, 0] > 0
        n_samples = ECE_mat[:, 0]
        mean_accuracy = ECE_mat[:, 1]
        mean_accuracy[ind] = ECE_mat[ind, 1].div(ECE_mat[ind, 0])
        mean_confidence = ECE_mat[:, 2]
        mean_confidence[ind] = ECE_mat[ind, 2].div(ECE_mat[ind, 0])
        return n_samples, mean_confidence, mean_accuracy

    
    def __call__(self, label_pred, tar_prob_pred, lds, 
                 measure_overconfidence=False, return_ECE_mat=False):
        ECE_mat = None
        with tc.no_grad():
            for ld in lds:
                for i, (xs, ys) in enumerate(ld):
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)
                    yhs = label_pred(xs)
                    phs = tar_prob_pred(xs, yhs)

                    ECE_mat_b = self.get_acc_conf_mat(yhs, phs, ys)
                    ECE_mat = ECE_mat + ECE_mat_b if ECE_mat is not None else ECE_mat_b

        if measure_overconfidence:
            ECE = self.ECEmat2ECE_overconfidence(ECE_mat)
        else:
            ECE = self.ECEmat2ECE(ECE_mat)
        if return_ECE_mat:
            return ECE, ECE_mat
        else:
            return ECE
    
    def plot_reliablity_diagram(self, fig_fn, label_pred, conf_pred, lds):
        ECE, ECE_mat = self(label_pred, conf_pred, lds, False, True)
        n_samples, mean_confidence, mean_accuracy = self.decomposeECEmat(ECE_mat)
        plot_reliability_diag(self.n_bins, mean_accuracy, n_samples, fig_fn=fig_fn, fontsize=20, 
                              ECE=ECE)

        
def plot_reliability_diag(n_bins, mean_accuracy, n_samples, fig_fn=None, fontsize=15, ECE=None):
    out_fn = fig_fn + '_conf_acc'
    with PdfPages(out_fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        plt.rc('xtick',labelsize=fontsize*0.75)
        plt.rc('ytick',labelsize=fontsize*0.75)

        xs_ori = tc.linspace(0, 1, n_bins+1)
        xs = xs_ori[0:-1] + (xs_ori[1:] - xs_ori[0:-1]) / 2.0
        w = (xs[1] - xs[0]) * 0.75

        plt.bar(xs.numpy(), mean_accuracy.numpy(), width=w, color='r', edgecolor='k')
        plt.plot(xs_ori.numpy(), xs_ori.numpy(), 'k--')
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.grid(True)
        plt.xlabel("Confidence", fontsize=fontsize)
        plt.ylabel("Accuracy", fontsize=fontsize)
        if ECE is not None:
            plt.title("ECE = %.2f%%"%(ECE*100), fontsize=fontsize)
        plt.savefig(out_fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

    out_fn = fig_fn + '_conf_acc_freq'
    with PdfPages(out_fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        plt.rc('xtick',labelsize=fontsize*0.75)
        plt.rc('ytick',labelsize=fontsize*0.75)
        
        plt.bar(xs.numpy(), n_samples.div(n_samples.sum()).numpy(), 
                width=w, color='r', edgecolor='k')
        plt.xlim([0, 1.0])
        plt.grid(True)
        plt.xlabel("Confidence", fontsize=fontsize)
        plt.ylabel("Sample Ratio", fontsize=fontsize)

        plt.savefig(out_fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')



def eval_print_ECE(lds, ld_names, model, target_model):
    
    ## calibration error for each dataset
    ECEs = []
    for ld, ld_name in zip(lds, ld_names):
        ECE = CalibrationError_L1()(model.label_pred, target_model.tar_prob_pred, [ld])
        print("# %s calibration error = %.2f%%"%(ld_name, ECE * 100.0))
        ECEs.append(ECE.unsqueeze(0))

    if len(lds) > 1:
        ## calibration error for combined datasets
        ECE_all = CalibrationError_L1()(model.label_pred, target_model.tar_prob_pred, lds)
        print("# Combined calibration error = %.2f%%"%(ECE_all * 100.0))

        ## average 
        ECEs = tc.cat(ECEs)
        print("# Average calibration error = %.2f%%"%(ECEs.mean() * 100.0))
    
    ## overconfience stats
    ECEs = []
    for ld, ld_name in zip(lds, ld_names):
        ECE = CalibrationError_L1()(model.label_pred, target_model.tar_prob_pred, [ld], True)
        print("# %s over-confident calibration error = %.2f%%"%(ld_name, ECE * 100.0))
        ECEs.append(ECE.unsqueeze(0))

    if len(lds) > 1:
        ## calibration error for combined datasets
        ECE_all = CalibrationError_L1()(model.label_pred, target_model.tar_prob_pred, lds, True)
        print("# Combined over-confident calibration error = %.2f%%"%(ECE_all * 100.0))

        ## average 
        ECEs = tc.cat(ECEs)
        print("# Average over-confident calibration error = %.2f%%"%(ECEs.mean() * 100.0))
    
