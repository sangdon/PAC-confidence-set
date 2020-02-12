import os, sys

import torch as tc
import numpy as np


def one_hot(ys, n_labels):
    ys_onehot = tc.zeros(ys.size(0), n_labels, device=ys.device).long()
    ys_onehot.scatter_(1, ys.view(-1, 1), 1)
    return ys_onehot

def compute_cls_error(lds, model, device, feature_map=None, loss_fn=None):
    with tc.no_grad():
        model.eval()
        n_error = 0.0
        n_total = 0.0
        
        for ld in lds:
            for xs, ys in ld:
                if hasattr(xs, "to"):
                    xs = xs.to(device)
                else:
                    assert(hasattr(xs[0], "to"))
                    xs = [x.to(device) for x in xs]
                ys = ys.to(device)
                if feature_map is None:
                    zs = xs
                else:
                    zs = feature_map(xs)
                n_total += float(ys.size(0))
                if loss_fn is None:
                    yhs = model.label_pred(zs)
                    
                    n_error += (ys != yhs).float().sum()
                else:
                    assert(feature_map is None)
                    fhs = model(zs)
                    n_error += loss_fn(fhs, ys).sum()

        return n_error/n_total, n_error, n_total

def compute_print_cls_error(ld, ld_name, model, device, feature_map=None, loss_fn=None):
    error, n_error, n_total = compute_cls_error([ld], model, device, feature_map, loss_fn)
    print("[%s] error = %d / %d = %.2f%%"%(ld_name, n_error, n_total, error*100.0))
    

class JointLoader:
    def __init__(self, *lds):
        self.lds = lds
        self.longest_iter_idx = np.argmax([len(ld) for ld in lds])
        
    def __iter__(self):
        self.iters = [iter(ld) for ld in self.lds]
        return self
    
    def __next__(self):
        out = []
        for i, it in enumerate(self.iters):
            try:
                xs, ys = next(it)
            except StopIteration:
                if i == self.longest_iter_idx:
                    raise StopIteration
                else:
                    self.iters[i] = iter(self.lds[i])
                    xs, ys = next(self.iters[i])
            out.append(xs)
            out.append(ys)
        # maintain the same batch size
        bs_min = min([a.size(0) for a in out])
        out = [o[:bs_min] for o in out]
        return out
