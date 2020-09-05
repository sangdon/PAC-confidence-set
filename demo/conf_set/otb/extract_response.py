import os, sys
import pickle

sys.path.append("../../../")
from data.otb import *
from forecasters import load_forecaster

import torch as tc


if __name__ == "__main__":
    dsld = loadOTB("../datasets/otb", 100, bb_format="xyxy")
    ld_names = ['val1', 'val2', 'test']
    lds = [dsld.val1, dsld.val2, dsld.test]
    root = "../datasets/otb_precomp"
    
    model = load_forecaster(None)
    model.cuda()
    model.eval()
    
    # extract response
    for ld_name, ld in zip(ld_names, lds):
        subroot = os.path.join(root, ld_name)
        os.makedirs(subroot, exist_ok=True)
        i = 0
        for xs, ys in ld:
            xs = [x.cuda() for x in xs]
            ys = ys.cuda()
            
            yhs, yhs_var = model(xs)
            ys = model.baseF.encode_bb(xs, ys, model.baseF.opts)

            for y, yh, yh_var in zip(ys, yhs, yhs_var):
                fn = os.path.join(subroot, "%d.pk"%(i))
                print(fn)
                yh = yh.detach().cpu()
                yh_var = yh_var.detach().cpu()
                y = y.detach().cpu()
                pickle.dump((yh, yh_var, y), open(fn, "wb"))
                i += 1

            
            
