import os, sys
import types
import time
import numpy as np
import itertools
import copy
import math
import shutil

import torch as tc
import torch.tensor as T

import matplotlib.ticker as ticker

sys.path.append("../../")
from classification.utils import *
from calibration.utils import *
from conf_set.conf_set import *
from conf_set.utils import *

from opts.options import BaseArgParser


##
## train confidence sets for each combination of parameters (n, eps, delta)
##
def train_test(params):
    rets = []
    params.train.exp_name = os.path.join(params.exp_name, "F")
    params.cal.exp_name = os.path.join(params.exp_name, "F_cal")
    if params.cs.no_acc:
        params.cal.exp_name += "_no_acc"
        
    for (n, eps, delta) in itertools.product(params.cs.n, params.cs.eps, params.cs.delta):
        ## initialize a confidence set
        F = params.load_forecaster(params)
        if params.cs.cond_thres:
            C = params.ConfSet_cond(F, eps, delta, n, params=params.cs)
        else:
            C = params.ConfSet(F, eps, delta, n)
        
        ## train the confidence set
        csc = ConfSetConstructor(params, C)
        flag = csc.train(params.dataset.ld.train, params.dataset.ld.val1_train, params.dataset.ld.val1, params.dataset.ld.val2)

        ## test the forecaster
        if params.task == "cls":
            compute_print_cls_error(params.dataset.ld.test, 'test', F, device=csc.device)
            eval_print_ECE([params.dataset.ld.test], ['test'], F, F)
        
        ## test the confidence set
        if flag:
            ##TODO: save results for plots
            _, cs_errors = csc.test([params.dataset.ld.test], ['test'])
        else:
            print("[failed] confidence set estimation failed: n may be too small.")
            cs_errors = [None]

        ## keep
        rets.append({"n": n, "eps": eps, "delta": delta, "model_CS": C, "cs_errors": cs_errors[0], "success": flag})

    return rets
    
##
## plot boxes for ablation study
##
def compute_conf_set_size_plot(params, res):
    ld = params.dataset.ld
    for r in res:
        if not r['success']:
            r['css'] = None
            continue

        eps = r['eps']
        delta = r['delta']
        n = r['n']
        model_CS = r['model_CS']
        
        ## compute confidenc set size
        csc = ConfSetConstructor(params, model_CS)
        css = csc.conf_set_size(ld.test, log_scale=True if params.task=="rl" else False, time_summary=params.plot.time_summary)

        ## summary trajectories
        if params.plot.ex_summary == 'none':
            # do not summary examples: used for box plots
            r['css'] = css
        else:
            assert(params.task == "rl")
            # summary along examples: used for trajectory plots
            ## confidence set size is in log-scale
            if params.plot.ex_summary == 'mean':
                # mean among examples
                n_examples = css.size(0)
                r['css'] = 0.5*(css - np.log(n_examples)).logsumexp(0)
            elif params.plot.ex_summary == 'median':
                # median among examples
                r['css'] = 0.5*css.median(0)[0] # take a square root
            else:
                raise NotImplementedError    
                        
    return res

def conf_set_summary_no_acc(params):
    params.cs.no_cal = False
    params.cs.no_acc = True
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = False

    return conf_set_summary(params)
    
def conf_set_summary_no_cal(params):
    params.cs.no_cal = True
    params.cs.no_acc = False
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = False

    return conf_set_summary(params)

def conf_set_summary_no_db(params):
    params.cs.no_cal = False
    params.cs.no_acc = False
    params.cs.no_db = True
    params.cs.no_pac = False
    params.cs.no_error = False

    return conf_set_summary(params)

def conf_set_summary_naive(params):
    params.cs.no_cal = False
    params.cs.no_acc = False
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = False
    params.cs.cond_thres = True

    return conf_set_summary(params)

def conf_set_summary_naive_no_cal(params):
    params.cs.no_cal = True
    params.cs.no_acc = False
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = False
    params.cs.cond_thres = True

    return conf_set_summary(params)

def conf_set_summary_no_error(params):
    params.cs.no_cal = False
    params.cs.no_acc = False
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = True

    return conf_set_summary(params)

def conf_set_summary_ours(params):
    params.cs.no_cal = False
    params.cs.no_acc = False
    params.cs.no_db = False
    params.cs.no_pac = False
    params.cs.no_error = False

    return conf_set_summary(params)

def conf_set_summary(params):
    ## load a confidence set
    res = train_test(params)
    ## compute confidence set sizes
    res = compute_conf_set_size_plot(params, res)
    return res

##
## plot procedures
##
def plot_conf_set_comp(params):
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_comp")
    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size and error
    labels = ["naive\n(no cal.)", "naive", "ours"]
    stats = [
        conf_set_summary_naive_no_cal(copy.deepcopy(params)),
        conf_set_summary_naive(copy.deepcopy(params)),
        conf_set_summary_ours(copy.deepcopy(params)),
    ]

    ## box plot
    plot_conf_set_box_(params, plot_root, stats, labels)

    ## cs error bar plot
    plot_conf_set_error(params, plot_root, stats, labels)

    
def plot_conf_set_error(params, plot_root, stats, cs_labels):
    os.makedirs(plot_root, exist_ok=True)

    ## trend of epsilon and delta
    for stat in zip(*stats):
        n = stat[0]['n']
        delta = stat[0]['delta']
        eps = stat[0]['eps']
        for s in stat:            
            assert(s['n'] == n and s['delta'] == delta and s['eps'] == eps)

        ## plot error
        cs_plot = tc.tensor([s['cs_errors'][0] for s in stat])
        
        ## test
        plot_fn = os.path.join(plot_root, "cs_error_n_%d_delta_%f_eps_%f.png"%(n, delta, eps))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()

            ## cs error
            plt.bar(range(1, len(cs_labels)+1), cs_plot.numpy(), width=0.75, color='r', edgecolor='k')
            ## eps
            h_eps = plt.plot(range(0, len(cs_labels)+2), [eps]*(len(cs_labels)+2), "k--", label=r"$\epsilon = 0.01$")
            
            ## beautify    
            plt.xticks(range(1, len(cs_labels)+1), cs_labels)
            plt.xticks(fontsize=int(params.plot.font_size*0.8))
            plt.yticks(fontsize=int(params.plot.font_size*0.8))
            plt.grid(True)
            plt.ylabel("Conf. set error", fontsize=params.plot.font_size)
            plt.legend(handles=[h_eps[0]], fontsize=params.plot.font_size)
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')

def plot_conf_set_box(params):
    if params.task == "cls":
        plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_box")
        shutil.rmtree(plot_root, ignore_errors=True)
        stats = [
            conf_set_summary_no_db(copy.deepcopy(params)),
            conf_set_summary_no_cal(copy.deepcopy(params)),
            conf_set_summary_ours(copy.deepcopy(params))
        ]
        labels = ["C", "D", "C+D"]
        plot_conf_set_box_(params, plot_root, stats, labels)
        
    elif params.task == "reg":
        plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_box")
        shutil.rmtree(plot_root, ignore_errors=True)
        stats = [
            conf_set_summary_no_db(copy.deepcopy(params)),
            conf_set_summary_no_cal(copy.deepcopy(params)),
            conf_set_summary_ours(copy.deepcopy(params))
        ]
        labels = ["C", "D", "C+D"]
        plot_conf_set_box_(params, plot_root, stats, labels)
        
    elif params.task == "rl":
        plot_conf_set_box_rl(params)
        
    else:
        raise NotImplementedError
            

def plot_conf_set_box_(params, plot_root=None, stats=None, cs_labels=None):

    assert(params.task != "rl")
    
    os.makedirs(plot_root, exist_ok=True)

    ## trend of epsilon and delta
    for stat in zip(*stats):
        n = stat[0]['n']
        delta = stat[0]['delta']
        eps = stat[0]['eps']
        for s in stat:            
            assert(s['n'] == n and s['delta'] == delta and s['eps'] == eps)
        
        cs_labels_plot = []
        cs_plot = []
        for l, s in zip(cs_labels, stat):
            if s['success']:
                cs_labels_plot.append(l)
                cs_plot.append(s['css'].unsqueeze(1).cpu())
        cs_plot = tc.cat(cs_plot, 1)
        
        ## add small random value to avoid matplotlib warning (producing invalid box plot)
        cs_plot += tc.rand_like(cs_plot)*1e-6

        ## test
        plot_fn = os.path.join(plot_root, "box_n_%d_delta_%f_eps_%f.png"%(n, delta, eps))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()
            
            plt.boxplot(cs_plot.numpy(), whis=np.inf,
                        boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))
            plt.gca().set_xticklabels(cs_labels_plot)
            plt.xticks(fontsize=int(params.plot.font_size*0.8))
            plt.yticks(fontsize=int(params.plot.font_size*0.8))

            # grid
            plt.ylim(params.plot.ylim)    
            if params.plot.log_scale:
                plt.minorticks_on()
                plt.gca().grid(which='major', linestyle='-', linewidth='0.5', color='black')
                plt.gca().grid(which='minor', axis='y',  linestyle=(0, (1, 5)), linewidth='0.5', color='black')
            
                plt.ylabel("Conf. set size (log-scale)", fontsize=params.plot.font_size)
                plt.yscale("log")
            else:
                plt.grid(True)
                plt.ylabel("Conf. set size", fontsize=params.plot.font_size)
                
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')

def plot_conf_set_box_rl(params):
    
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_box")
    shutil.rmtree(plot_root, ignore_errors=True)
    if params.plot.symlog_scale:
        plot_root += "_symlog"
    elif params.plot.log_scale:
        plot_root += "_log"
    if len(params.plot.time_summary) == 1 and params.plot.time_summary[0] is not -1:
        plot_root += "_%d"%(params.plot.time_summary[0])

    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats_CD = conf_set_summary_no_acc(copy.deepcopy(params))
    stats_AD = conf_set_summary_no_cal(copy.deepcopy(params))
    stats_AC = conf_set_summary_no_db(copy.deepcopy(params))
    stats_ACD = conf_set_summary_ours(copy.deepcopy(params))

    ## trend of epsilon and delta
    for stat_CD, stat_AD, stat_AC, stat_ACD in zip(stats_CD, stats_AD, stats_AC, stats_ACD):
        n = stat_ACD['n']
        delta = stat_ACD['delta']
        eps = stat_ACD['eps']
        assert(stat_CD['n'] == n and stat_CD['delta'] == delta and stat_CD['eps'] == eps)
        assert(stat_AD['n'] == n and stat_AD['delta'] == delta and stat_AD['eps'] == eps)
        assert(stat_AC['n'] == n and stat_AC['delta'] == delta and stat_AC['eps'] == eps)
        
        ## draw box plots
        if stat_AC['css'] is None:
            cs_labels = ['C+D', "A+D", "A+C+D"]
            cs_plot = tc.cat((
                stat_CD['css'].unsqueeze(1),
                stat_AD['css'].unsqueeze(1),
                stat_ACD['css'].unsqueeze(1)), 1).cpu()
        else:
            cs_labels = ['C+D', "A+D", "A+C", "A+C+D"]
            cs_plot = tc.cat((
                stat_CD['css'].unsqueeze(1),
                stat_AD['css'].unsqueeze(1),
                stat_AC['css'].unsqueeze(1),
                stat_ACD['css'].unsqueeze(1)), 1).cpu()
            
        if params.task == "rl":
            cs_plot = cs_plot.exp()

        ## test
        plot_fn = os.path.join(plot_root, "ablation_n_%d_delta_%f_eps_%f.png"%(n, delta, eps))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()

            if params.plot.symlog_scale:
                # symlog y scale
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))

                # tick font size
                plt.xticks(fontsize=int(params.plot.font_size*0.8))
                plt.yticks(fontsize=int(params.plot.font_size*0.8))

                # labels
                plt.gca().set_xticklabels(cs_labels)
                plt.ylabel("Conf. set size (linear/log)", fontsize=params.plot.font_size)

                # yscale
                plt.ylim(params.plot.ylim[0], params.plot.ylim[2])
                plt.yscale("symlog", linthreshy=params.plot.ylim[1], linscaley=math.log10(params.plot.ylim[2])/1.0)
                plt.gca().yaxis.set_major_locator(ticker.FixedLocator(
                    np.arange(params.plot.ylim[0], params.plot.ylim[1], (params.plot.ylim[1]-params.plot.ylim[0])//5).tolist()+
                    [float(10**i) for i in range(
                        round(math.log10(params.plot.ylim[1])),
                        round(math.log10(params.plot.ylim[2])),
                        (round(math.log10(params.plot.ylim[2]))-round(math.log10(params.plot.ylim[1])))//5
                    )]
                ))
                def x_fmt(x, pos):
                    if x == 0:
                        return r'$0$'
                    elif x/(10**math.ceil(math.log(x, 10))) == 1:
                        return r'$10^{%d}$'%(math.ceil(math.log(x, 10)))
                    else:
                        return r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))
                
                # beautify
                plt.grid(True)

                
            else:
                # linear or log scale
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))
                plt.gca().set_xticklabels(cs_labels)
                plt.xticks(fontsize=int(params.plot.font_size*0.8))
                plt.yticks(fontsize=int(params.plot.font_size*0.8))
                plt.grid(True)

                plt.ylim(params.plot.ylim)    
                if params.plot.log_scale:
                    plt.ylabel("Conf. set size (log-scale)", fontsize=params.plot.font_size)
                    plt.yscale("log")
                else:
                    plt.ylabel("Conf. set size", fontsize=params.plot.font_size)
                
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')
            plt.close()

##
## plot summary trajectories
##          
def plot_conf_set_traj(params):
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_traj_%s"%(params.plot.ex_summary))
    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats_CD = conf_set_summary_no_acc(copy.deepcopy(params))
    stats_AD = conf_set_summary_no_cal(copy.deepcopy(params))
    stats_AC = conf_set_summary_no_db(copy.deepcopy(params))
    stats_ACD = conf_set_summary_ours(copy.deepcopy(params))

    ## trend of epsilon and delta
    for stat_CD, stat_AD, stat_AC, stat_ACD in zip(stats_CD, stats_AD, stats_AC, stats_ACD):
        n = stat_ACD['n']
        delta = stat_ACD['delta']
        eps = stat_ACD['eps']
        assert(stat_CD['n'] == n and stat_CD['delta'] == delta and stat_CD['eps'] == eps)
        assert(stat_AD['n'] == n and stat_AD['delta'] == delta and stat_AD['eps'] == eps)
        assert(stat_AC['n'] == n and stat_AC['delta'] == delta and stat_AC['eps'] == eps)

        ## draw trajectories
        fn = os.path.join(plot_root, "ablation_n_%d_delta_%f_eps_%f.png"%(n, delta, eps))
        with PdfPages(fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()
            hs = []
            ## C+D
            color = "purple"
            css_plot = stat_CD['css'].cpu().double().exp().numpy()
            h_CD = plt.plot(range(1, params.dataset.n_steps+1), css_plot, "x-", color=color, label="C+D")
            print("C+D:", css_plot)
            hs.append(h_CD[0])
            
            ## A+D
            color = "blue"
            css_plot = stat_AD['css'].cpu().double().exp().numpy()
            h_AD = plt.plot(range(1, params.dataset.n_steps+1), css_plot, "+-", color=color, label="A+D")
            print("A+D:", css_plot)
            hs.append(h_AD[0])
            
            ## A+C
            if stat_AC['css'] is not None:
                color = "green"
                css_plot = stat_AC['css'].cpu().double().exp().numpy()
                h_AC = plt.plot(range(1, params.dataset.n_steps+1), css_plot, "o-", color=color, label="A+C")
                print("A+C:", css_plot)
                hs.append(h_AC[0])
            
            
            ## A+C+D
            color = "red"
            css_plot = stat_ACD['css'].cpu().double().exp().numpy()
            h_ACD = plt.plot(range(1, params.dataset.n_steps+1), css_plot, "s-", color=color, label="A+C+D (ours)")
            print("A+C+D:", css_plot)
            hs.append(h_ACD[0])
            
            ## beautify
            plt.grid(True)
            plt.xlabel("Time step", fontsize=params.plot.font_size)
            plt.ylabel("Conf. set size (log-scale)", fontsize=params.plot.font_size)
            plt.ylim(params.plot.ylim)
            plt.yscale('log')
            plt.rc('xtick', labelsize=params.plot.font_size*0.75)
            plt.rc('ytick', labelsize=params.plot.font_size*0.75)
            plt.legend(handles=hs, fontsize=params.plot.font_size)
            
            plt.savefig(fn, bbox_inches="tight")
            pdf.savefig(bbox_inches="tight")


            
##
## plot a epsilon-dependency plot
##          
def plot_conf_set_eps(params):
    
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_eps")
    if params.plot.broken:
        from matplotlib.ticker import FuncFormatter
        plot_root += "_broken"
    elif params.plot.symlog_scale:
        plot_root += "_symlog"
    elif params.plot.log_scale:
        plot_root += "_log"

    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats = conf_set_summary_ours(copy.deepcopy(params))
    stats_eps = np.array([s['eps'] for s in stats])
    stats_delta = np.array([s['delta'] for s in stats])
    stats_n = np.array([s['n'] for s in stats])
    stats_css = tc.cat([s['css'].unsqueeze(0) for s in stats], 0)

    for delta, n in itertools.product(params.cs.delta, params.cs.n):
        idx = (stats_delta == delta) & (stats_n == n)
        
        ## draw a box plot
        cs_labels = [r"$%.2f$"%(e) for e in params.cs.eps]
        cs_plot = tc.cat([stats_css[tc.tensor((idx & (stats_eps == e)).tolist())].squeeze().unsqueeze(1) for e in params.cs.eps], 1).cpu()
        
        if params.task == "rl":
            cs_plot = cs_plot.exp()

        ## test
        plot_fn = os.path.join(plot_root, "eps_n_%d_delta_%f.png"%(n, delta))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()

            if params.plot.broken:
                fig, (ax_up, ax_down) = plt.subplots(2, 1)
            
                ## upper
                res_up = ax_up.boxplot(cs_plot.numpy(), whis=np.inf,
                                       boxprops=dict(linewidth=0), medianprops=dict(linewidth=0), flierprops=dict(markersize=0))
                ## lower
                ax_down.boxplot(cs_plot.numpy(), whis=np.inf,
                                boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))

                ## remove lower cap
                for i in range(0, len(res_up['caps']), 2):
                    cap = res_up['caps'][i]
                    cap.set(xdata=cap.get_xdata()*0.0)
            
                ## ylim
                ax_up.set_ylim(ymin=params.plot.ylim[2], ymax=params.plot.ylim[3])
                ax_down.set_ylim(ymin=params.plot.ylim[0], ymax=params.plot.ylim[1])

                ## remove ticks
                ax_up.spines['bottom'].set_visible(False)
                ax_up.tick_params(bottom=False, labelbottom=False)
                ax_down.spines['top'].set_visible(False)

                ## tick labels
                ax_down.set_xticklabels(cs_labels, fontsize=int(params.plot.font_size*0.8))
                ax_up.yaxis.set_major_formatter(FuncFormatter(
                    lambda x, pos: r'$0$' if x == 0 else r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))))
                ax_down.yaxis.set_major_formatter(FuncFormatter(
                    lambda x, pos: r'$0$' if x == 0 else r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))))
                ax_up.yaxis.set_tick_params(labelsize=int(params.plot.font_size*0.6))
                ax_down.yaxis.set_tick_params(labelsize=int(params.plot.font_size*0.6))
                
                ## axis lables
                plt.xlabel(r"$\epsilon$", fontsize=params.plot.font_size)
                fig.text(-0.05, 0.5, "Conf. set size", fontsize=params.plot.font_size, va='center', rotation='vertical')
                
                ## diagonal
                d = .015
                kwargs = dict(transform=ax_up.transAxes, color='k', clip_on=False)
                ax_up.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                ax_up.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax_down.transAxes)  # switch to the bottom axes
                ax_down.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax_down.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            
                ## beautify
                ax_up.grid(True)
                ax_down.grid(True)

            elif params.plot.symlog_scale:
                # symlog y scale
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))

                # tick font size
                plt.xticks(fontsize=int(params.plot.font_size*0.8))
                plt.yticks(fontsize=int(params.plot.font_size*0.8))

                # labels
                plt.xlabel(r"$\epsilon$", fontsize=params.plot.font_size)
                plt.gca().set_xticklabels(cs_labels)
                plt.ylabel("Conf. set size (linear/log)", fontsize=params.plot.font_size)

                # yscale
                plt.ylim(params.plot.ylim[0], params.plot.ylim[2])
                plt.yscale("symlog", linthreshy=params.plot.ylim[1], linscaley=math.log10(params.plot.ylim[2])/1.0)
                plt.gca().yaxis.set_major_locator(ticker.FixedLocator(
                    np.arange(params.plot.ylim[0], params.plot.ylim[1], (params.plot.ylim[1]-params.plot.ylim[0])//5).tolist()+
                    [float(10**i) for i in range(
                        math.ceil(math.log10(params.plot.ylim[1])),
                        round(math.log10(params.plot.ylim[2])),
                        (round(math.log10(params.plot.ylim[2]))-math.ceil(math.log10(params.plot.ylim[1])))//5
                    )]
                ))
                def x_fmt(x, pos):
                    if x == 0:
                        return r'$0$'
                    elif x/(10**math.ceil(math.log(x, 10))) == 1:
                        return r'$10^{%d}$'%(math.ceil(math.log(x, 10)))
                    else:
                        return r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))
                
                # beautify
                plt.grid(True)

            else:
                ## no-broken axis
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))
                plt.gca().set_xticklabels(cs_labels)
                plt.xticks(fontsize=int(params.plot.font_size*0.75))
                plt.yticks(fontsize=int(params.plot.font_size*0.75))
                plt.grid(True)
                
                plt.xlabel(r"$\epsilon$", fontsize=params.plot.font_size)
                plt.ylim(params.plot.ylim)    
                if params.plot.log_scale:
                    plt.ylabel("Conf. set size (log-scale)", fontsize=params.plot.font_size)
                    plt.yscale("log")
                else:
                    plt.ylabel("Conf. set size", fontsize=params.plot.font_size)
                
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')
            plt.close()
        
##
## plot a delta-dependency plot
##          
def plot_conf_set_delta(params):
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_delta")
    if params.plot.broken:
        from matplotlib.ticker import FuncFormatter
        plot_root += "_broken"
    elif params.plot.symlog_scale:
        plot_root += "_symlog"
    elif params.plot.log_scale:
        plot_root += "_log"

    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats = conf_set_summary_ours(copy.deepcopy(params))
    stats_eps = np.array([s['eps'] for s in stats])
    stats_delta = np.array([s['delta'] for s in stats])
    stats_n = np.array([s['n'] for s in stats])
    stats_css = tc.cat([s['css'].unsqueeze(0) for s in stats], 0)

    for eps, n in itertools.product(params.cs.eps, params.cs.n):
        idx = (stats_eps == eps) & (stats_n == n)
        
        ## draw a box plot
        cs_labels = [r"$10^{%d}$"%(round(math.log(d, 10))) for d in params.cs.delta]
        cs_plot = tc.cat([stats_css[tc.tensor((idx & (stats_delta == d)).tolist())].squeeze().unsqueeze(1) for d in params.cs.delta], 1).cpu()

        if params.task == "rl":
            cs_plot = cs_plot.exp()

        ## test
        plot_fn = os.path.join(plot_root, "delta_n_%d_eps_%f.png"%(n, eps))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()

            if params.plot.broken:
                fig, (ax_up, ax_down) = plt.subplots(2, 1)
            
                ## upper
                res_up = ax_up.boxplot(cs_plot.numpy(), whis=np.inf,
                                       boxprops=dict(linewidth=0), medianprops=dict(linewidth=0), flierprops=dict(markersize=0))
                ## lower
                ax_down.boxplot(cs_plot.numpy(), whis=np.inf,
                                boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))

                ## remove lower cap
                for i in range(0, len(res_up['caps']), 2):
                    cap = res_up['caps'][i]
                    cap.set(xdata=cap.get_xdata()*0.0)
            
                ## ylim
                ax_up.set_ylim(ymin=params.plot.ylim[2], ymax=params.plot.ylim[3])
                ax_down.set_ylim(ymin=params.plot.ylim[0], ymax=params.plot.ylim[1])

                ## remove ticks
                ax_up.spines['bottom'].set_visible(False)
                ax_up.tick_params(bottom=False, labelbottom=False)
                ax_down.spines['top'].set_visible(False)

                ## tick labels
                ax_down.set_xticklabels(cs_labels, fontsize=int(params.plot.font_size*0.8))
                ax_up.yaxis.set_major_formatter(FuncFormatter(
                    lambda x, pos: r'$0$' if x == 0 else r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))))
                ax_down.yaxis.set_major_formatter(FuncFormatter(
                    lambda x, pos: r'$0$' if x == 0 else r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))))
                ax_up.yaxis.set_tick_params(labelsize=int(params.plot.font_size*0.6))
                ax_down.yaxis.set_tick_params(labelsize=int(params.plot.font_size*0.6))
                
                ## axis lables
                plt.xlabel(r"$\delta$", fontsize=params.plot.font_size)
                fig.text(-0.05, 0.5, "Conf. set size", fontsize=params.plot.font_size, va='center', rotation='vertical')
                
                ## diagonal
                d = .015
                kwargs = dict(transform=ax_up.transAxes, color='k', clip_on=False)
                ax_up.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                ax_up.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax_down.transAxes)  # switch to the bottom axes
                ax_down.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax_down.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            
                ## beautify
                ax_up.grid(True)
                ax_down.grid(True)
                
            elif params.plot.symlog_scale:
                # symlog y scale
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))

                # tick font size
                plt.xticks(fontsize=int(params.plot.font_size*0.8))
                plt.yticks(fontsize=int(params.plot.font_size*0.8))

                # labels
                plt.gca().set_xticklabels(cs_labels)
                plt.xlabel(r"$\delta$", fontsize=params.plot.font_size)
                plt.ylabel("Conf. set size (linear/log)", fontsize=params.plot.font_size)

                # yscale
                plt.ylim(params.plot.ylim[0], params.plot.ylim[2])
                plt.yscale("symlog", linthreshy=params.plot.ylim[1], linscaley=math.log10(params.plot.ylim[2])/1.0)
                plt.gca().yaxis.set_major_locator(ticker.FixedLocator(
                    np.arange(params.plot.ylim[0], params.plot.ylim[1], (params.plot.ylim[1]-params.plot.ylim[0])//5).tolist()+
                    [float(10**i) for i in range(
                        math.ceil(math.log10(params.plot.ylim[1])),
                        round(math.log10(params.plot.ylim[2])),
                        (round(math.log10(params.plot.ylim[2]))-math.ceil(math.log10(params.plot.ylim[1])))//5
                    )]
                ))
                def x_fmt(x, pos):
                    if x == 0:
                        return r'$0$'
                    elif x/(10**math.ceil(math.log(x, 10))) == 1:
                        return r'$10^{%d}$'%(math.ceil(math.log(x, 10)))
                    else:
                        return r'$%1.2fx10^{%d}$'%(x/(10**math.ceil(math.log(x, 10))), math.ceil(math.log(x, 10)))
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))
                
                # beautify
                plt.grid(True)

            else:
            
                plt.boxplot(cs_plot.numpy(), whis=np.inf,
                            boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=2))
                plt.gca().set_xticklabels(cs_labels)
                plt.xticks(fontsize=int(params.plot.font_size*0.75))
                plt.yticks(fontsize=int(params.plot.font_size*0.75))
                plt.grid(True)
            
                plt.xlabel(r"$\delta$", fontsize=params.plot.font_size)
                plt.ylim(params.plot.ylim)    
                if params.plot.log_scale:
                    plt.ylabel("Conf. set size (log-scale)", fontsize=params.plot.font_size)
                    plt.yscale("log")
                else:
                    plt.ylabel("Conf. set size", fontsize=params.plot.font_size)
                
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')
            plt.close()
    

##
## conditional confidence set size
##
def plot_conf_set_cond(params):

    assert(params.task == "cls")
    
    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_cond")
    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats = conf_set_summary_ours(copy.deepcopy(params))
    stats_eps = np.array([s['eps'] for s in stats])
    stats_delta = np.array([s['delta'] for s in stats])
    stats_n = np.array([s['n'] for s in stats])
    stats_css = tc.cat([s['css'].unsqueeze(0) for s in stats], 0)
    
    for eps, delta, n in itertools.product(params.cs.eps, params.cs.delta, params.cs.n):
        idx = (stats_eps == eps) & (stats_n == n) & (stats_delta == delta)
        assert(np.sum(idx) == 1)
        for i, f in enumerate(idx):
            if f:
                model_CS = stats[i]['model_CS']
                model_F = model_CS.model_F
                break
        
        ## check correct images and confidence set size
        css_correct = []
        css_incorrect = []
        for xs, ys in params.dataset.ld.test:
            if params.use_gpu:
                xs = xs.cuda()
                ys = ys.cuda()
            with tc.no_grad():
                css = model_CS.size(xs)
                yhs = model_F.label_pred(xs)
                idx_corr = ys == yhs
                css_correct.append(css[idx_corr])
                css_incorrect.append(css[~idx_corr])
        css_correct = tc.cat(css_correct)
        css_incorrect = tc.cat(css_incorrect)
        print(css_correct.size())
        print(css_incorrect.size())

        ## box plot
        css_plot = (css_correct.unsqueeze(1).cpu().numpy(), css_incorrect.cpu().numpy())
        cs_labels = ["correct", "incorrect"]
        plot_fn = os.path.join(plot_root, "cond_n_%d_delta_%f_eps_%f.png"%(n, delta, eps))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()
            plt.boxplot(css_plot, whis=np.inf,
                        boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
            plt.gca().set_xticklabels(cs_labels)
            plt.xticks(fontsize=int(params.plot.font_size*0.75))
            plt.yticks(fontsize=int(params.plot.font_size*0.75))
            plt.grid(True)
            
            plt.ylim(params.plot.ylim)
            plt.ylabel("Conf. set size", fontsize=params.plot.font_size)
            
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')

##
## cls examples
##
def plot_exs(params):
    import torchvision.transforms as tform
    assert(params.task == "cls")
    ConfSetModel = ConfSetCls
    from imagenet.forecasters import load_forecaster
    F = load_forecaster(params)
    cs_rng = [1, 5, 10, 20]
    device = tc.device("cuda:0") if params.use_gpu else tc.device("cpu")
    fig_root = os.path.join(params.snapshot_root, params.exp_name, "fig_exs")
    os.makedirs(fig_root, exist_ok=True)

    ## load a model
    load_root = os.path.join(params.snapshot_root, params.exp_name)
    for n, eps, delta in itertools.product(params.cs.n, params.cs.eps, params.cs.delta):
        cs_model = ConfSetModel(F, eps, delta, n, load_root)
        cs_model = cs_model.to(device)

        ## iterates
        imgs_cond = []
        cs_cond = []
        css_cond = []
        labels_cond = []
        ys_cond = []
        yhs_cond = []
        for (xs, ys), (xs_ori, ys_ori) in zip(params.dataset.ld.test, params.dataset.ld_ori.test):

            xs = xs.to(device)
            ys = ys.to(device)
            ys_ori = ys_ori.to(device)
            
            assert(all(ys == ys_ori))
            ## compute cs and css
            cs_i = cs_model(xs)
            css_i = cs_model.size(xs)
            ys_i = ys
            yhs_i = cs_model.model_F.label_pred(xs)
            
            for i in range(0, len(cs_rng)-1):
                idx = (css_i >= cs_rng[i]) & (css_i < cs_rng[i+1])
                if any(idx):
                    imgs_cond.append(xs_ori[idx])
                    cs_cond.append(cs_i[idx])
                    css_cond.append(css_i[idx])
                    ys_cond.append(ys_i[idx])
                    yhs_cond.append(yhs_i[idx])
        imgs_cond = tc.cat(imgs_cond, 0)
        cs_cond = tc.cat(cs_cond, 0)
        css_cond = tc.cat(css_cond)
        ys_cond = tc.cat(ys_cond)
        yhs_cond = tc.cat(yhs_cond)
        
        ## write
        for i in range(0, len(cs_rng)-1):
            fig_subroot = os.path.join(fig_root, "css_range_%d_%d"%(cs_rng[i], cs_rng[i+1]))
            os.makedirs(fig_subroot, exist_ok=True)
            for img_id, (img, cs, css, y, yh) in enumerate(zip(imgs_cond, cs_cond, css_cond, ys_cond, yhs_cond)):
                if css >= cs_rng[i] and css < cs_rng[i+1]:
                    fn = os.path.join(fig_subroot, "id_%d_css_%d.png"%(img_id, css))
                    # save an image
                    im = tform.ToPILImage()(img)
                    im.save(fn, "png")
                    # save a confidence set
                    fn_cs = fn + ".txt"
                    cs_label = []
                    for label_idx, cs_i in enumerate(cs):
                        if cs_i == 1:
                            label = params.dataset.ld.names[label_idx]
                            label = "\\text{%s}"%(label)
                            if y == label_idx:
                                label = "{\\color{red}%s}"%(label)
                            if yh == label_idx:
                                label = "\\widehat{%s}"%(label)
                            cs_label.append(label)
                    cs_str = ", ".join(cs_label)
                    cs_str = cs_str.replace("_", " ")
                    print(cs_str)
                    with open(fn_cs, "w") as f:
                        f.write(cs_str)

def plot_confexs(params):

    plot_root = os.path.join(params.snapshot_root, params.exp_name, "plot_confexs")
    shutil.rmtree(plot_root, ignore_errors=True)
    os.makedirs(plot_root, exist_ok=True)

    ## compute confidence set size
    stats = conf_set_summary(copy.deepcopy(params))
    stats_eps = np.array([s['eps'] for s in stats])
    stats_delta = np.array([s['delta'] for s in stats])
    stats_n = np.array([s['n'] for s in stats])
    model_CSs = [s['model_CS'] for s in stats]
    model_Fs = [m.model_F for m in model_CSs]
    assert(len(set(stats_n)) == 1)
    assert(len(set(stats_delta)) == 1)
    n = stats_n[0]
    delta = stats_delta[0]
    
    ## computation may redundent
    labels = []

    n_totals = []
    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []
    for eps, model_CS, model_F in zip(stats_eps, model_CSs, model_Fs):

        n_total = 0.0
        n_conf_exs = 0.0
        n_pos = 0.0
        n_neg = 0.0
        n_pos_exs = 0.0
        n_neg_exs = 0.0
        n_pos_corr_exs = 0.0
        n_neg_corr_exs = 0.0
        n_fapred_pos_exs = 0.0

        n_supp_pos = 0.0
        n_supp_neg = 0.0
        
        for xs, ys in params.dataset.ld.test:
            if params.use_gpu:
                xs = xs.cuda()
                ys = ys.cuda()
            with tc.no_grad():
                # predictions
                css = model_CS.size(xs)
                yhs = model_F.label_pred(xs)
                
                idx_conf_exs = css == 1
                idx_corr = ys == yhs
                idx_pos = ys == 1
                idx_neg = ys == 0

                
                n_total += ys.size(0)
                n_pos += (ys == 1).sum()
                n_neg += (ys == 0).sum()
                n_conf_exs += idx_conf_exs.sum()
                n_pos_exs += (idx_conf_exs & idx_pos).sum()
                n_neg_exs += (idx_conf_exs & idx_neg).sum()
                n_pos_corr_exs += (idx_conf_exs & idx_pos & (yhs==1)).sum()
                n_neg_corr_exs += (idx_conf_exs & idx_neg & (yhs==0)).sum()
                n_fapred_pos_exs += (idx_conf_exs & idx_pos & (yhs==0)).sum()


                
        print("# confident exs: %d/%d = %f"%(n_conf_exs, n_total, n_conf_exs/n_total))
        print("# (false alarm pred) / (singleton & actionable alarms): %d/%d = %f (=0?) "%(n_fapred_pos_exs, n_pos_exs, n_fapred_pos_exs/n_pos_exs))
        print("# (false alarm pred) / (singleton & false alarm): %d/%d = %f (=1?)"%(n_neg_corr_exs, n_neg_exs, n_neg_corr_exs/n_neg_exs))
        print("# suppressed true alarms: %d/%d = %f (=0?) "%(n_fapred_pos_exs, n_pos, n_fapred_pos_exs/n_pos))
        print("# suppressed false alarms: %d/%d = %f (=1?)"%(n_neg_corr_exs, n_neg, n_neg_corr_exs/n_neg))

        ## plot 
        plot_fn = os.path.join(plot_root, "sup_bar_n_%d_eps_%f_delta_%f.png"%(n, eps, delta))
        with PdfPages(plot_fn + '.pdf') as pdf: 
            plt.figure(1)
            plt.clf()
            labels_plot = ['false alarm', "actionable alarm"]
            ## stacked bar plots
            barwidth = 0.20

            ## suppsed ratio
            plt.bar([0, 1], [n_neg_corr_exs/n_neg, n_fapred_pos_exs/n_pos], width=barwidth, color="red", tick_label=labels_plot)

            # beautify
            #plt.xticks([0, 1], labels_plot, fontsize=int(params.plot.font_size*0.75))
            plt.xticks(fontsize=int(params.plot.font_size*0.75))
            plt.yticks(fontsize=int(params.plot.font_size*0.75))
            plt.grid(True)
            plt.ylabel("suppressed alarm ratio (%)", fontsize=params.plot.font_size)
            # save
            plt.savefig(plot_fn, bbox_inches='tight')
            pdf.savefig(bbox_inches='tight')
            plt.close()
            

        ## collect stats
        labels.append(r"$\epsilon = %.2f$"%(eps))
        n_totals.append(n_total)
        true_pos.append(n_pos_corr_exs.cpu().item())
        true_neg.append(n_neg_corr_exs.cpu().item())
        false_pos.append(n_pos_exs.cpu().item() - n_pos_corr_exs.cpu().item())
        false_neg.append(n_neg_exs.cpu().item() - n_neg_corr_exs.cpu().item())
        
    ## plot confexs ratio
    plot_fn = os.path.join(plot_root, "stacked_bar_n_%d_delta_%f.png"%(n, delta))
    with PdfPages(plot_fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()

        n_totals = np.array(n_totals)
        true_pos = np.array(true_pos)
        true_neg = np.array(true_neg)
        false_pos = np.array(false_pos)
        false_neg = np.array(false_neg)
        
        ## stacked bar plots
        n_groups = len(labels)
        barwidth = 0.35
        # false negative rate
        false_neg_rate = np.divide(false_neg, n_totals)
        h1 = plt.bar(range(0, n_groups), false_neg_rate, width=barwidth, label="false negative rate")
        # true positive rate
        true_pos_rate = np.divide(true_pos, n_totals)
        h2 = plt.bar(range(0, n_groups), true_pos_rate, width=barwidth, bottom=false_neg_rate, label="true positive rate")
        # false positive rate
        false_pos_rate = np.divide(false_pos, n_totals)
        h3 = plt.bar(range(0, n_groups), false_pos_rate, width=barwidth, bottom=false_neg_rate+true_pos_rate, label="false positive rate")
        # true negative rate
        true_neg_rate = np.divide(true_neg, n_totals)
        h4 = plt.bar(range(0, n_groups), true_neg_rate, width=barwidth, bottom=false_neg_rate+true_pos_rate+false_pos_rate, label="true negative rate")

        # beautify
        plt.xticks(range(0, n_groups), labels, fontsize=int(params.plot.font_size*0.75))
        plt.yticks(fontsize=int(params.plot.font_size*0.75))
        plt.grid(True)
        plt.ylabel("ratio of confident ex. (%)", fontsize=params.plot.font_size)
        plt.legend(handles=[h1, h2, h3, h4])
        # save
        plt.savefig(plot_fn, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        
if __name__ == "__main__":
    
    ## read user inputs
    params = BaseArgParser().read_args(os.path.splitext(os.path.basename(__file__))[0])

    if params.train_cs:
        ## train confidence sets
        train_test(params)
    else:
        ## plot results
        if params.plot.comp:
            plot_conf_set_comp(params)
        if params.plot.box:
            plot_conf_set_box(params)
        if params.plot.traj:
            plot_conf_set_traj(params)
        if params.plot.eps:
            plot_conf_set_eps(params)
        if params.plot.delta:
            plot_conf_set_delta(params)
        if params.plot.cond:
            plot_conf_set_cond(params)
        if params.plot.exs:
            plot_exs(params)
        if params.plot.confexs:
            plot_confexs(params)
        
