import os, sys
import pickle
from PIL import Image, ImageDraw

import torch as tc
from torchvision import transforms

sys.path.append("../../")
sys.path.append("otb")

from data.otb import *
from otb.forecasters import load_forecaster_full, load_forecaster

from conf_set.conf_set import ConfSetReg as ConfSetModel

def xyxy2xywh(xyxy):
    xywh = xyxy.clone()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh


if __name__ == "__main__":
    ## parameters
    dsld = loadOTB("datasets/otb", 100, bb_format="xyxy")
    cs_root = "otb/snapshots/pac_conf_set"
    plot_root = "otb/plot_conf_set"
    
    n = 5000
    eps = 0.01
    delta = 1e-5
    n_plots = float('inf')
    
    ## init
    os.makedirs(plot_root, exist_ok=True)
    
    # load a forecaster
    F_precomp = load_forecaster(None)
    assert(F_precomp.load(os.path.join(cs_root, "F_cal/model_params_cal")))
    print(F_precomp)
    
    F = load_forecaster_full()
    F.T.data = F_precomp.T
    print(F.cal_parameters())
    
    F.cuda()
    F.eval()

    print(F)
    
    ## load a confidence set
    C_precomp = ConfSetModel(F_precomp, eps, delta, n)
    C_precomp.load_cs(cs_root, 'cs') ## main results    
    
    C = ConfSetModel(F, eps, delta, n)
    C.T = C_precomp.T    
    C.eval()
    print(C.T)

    ## predict a confidence set for tracking
    i = 0
    for xs, ys in dsld.tracking:
        seq_ids = xs[-2]
        frame_ids = xs[-1]
        xs = [xs[0], xs[1], xs[4]] 
        xs = [x.cuda() for x in xs]
        ys = ys.cuda()

        ## original prediction
        yhs_ori, _ = C.model_F.baseF(xs)
        opts = C.model_F.baseF.opts
        yhs_ori = C.model_F.baseF.decode_output(xs, yhs_ori, opts)
        
        ## cs prediction
        lb, ub = C(xs)
        bb_ovap = tc.cat((lb[:, :2], ub[:, 2:]), 1)

        opts = C.model_F.baseF.opts
        yhs_cs_ovap = C.model_F.baseF.decode_output(xs, bb_ovap, opts)

        for x, y, seq_id, frame_id, y_ori, y_cs in zip(xs[2].detach().cpu(), ys.detach().cpu(), seq_ids, frame_ids, yhs_ori.detach().cpu(), yhs_cs_ovap.detach().cpu()):
            x_pil = transforms.ToPILImage()(x)
            
            draw = ImageDraw.Draw(x_pil)
            draw.rectangle(y.tolist(), outline="white", width=2)
            draw.rectangle(y_ori.tolist(), outline="red", width=2)
            draw.rectangle(y_cs.tolist(), outline="green", width=5) 

            fn = os.path.join(plot_root, "%s_%.4d.png"%(seq_id, frame_id))
            x_pil.save(fn)
            
            i += 1
            if i > n_plots:
                break
        if i > n_plots:
            break
        
            
