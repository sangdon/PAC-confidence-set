import os, sys

def compute_reg_loss(ld, model, loss_fn, device):
    n_total = 0.0
    loss = 0.0
    for xs, ys in ld:
        xs = xs.to(device)
        ys = ys.to(device)
        mus, logvars = model(xs)
        loss += loss_fn(mus, logvars, ys) * xs.size(0)
        n_total += xs.size(0)
    return loss/n_total, loss, n_total
