import os, sys
import math

import torch as tc
import torch.nn as nn
import torch.tensor as T
from torchvision import models
from torchvision import transforms

sys.path.append("../../../")
from models.BaseForecasters import Forecaster

from .boundingbox import BoundingBox
from .helper import crop_sample, crop_sample_bb, Rescale
from .eval import *

def load_forecaster(params):

    # ##FIXME: merge GoNet_var and GOTURNTracker
    # baseF = GoNet_var()
    # ##FIXME
    # checkpoint = tc.load("otb/pytorch_goturn.pth.tar", map_location=lambda storage, loc: storage)
    # #checkpoint = tc.load("pytorch_goturn.pth.tar", map_location=lambda storage, loc: storage) ##FIXME
    # baseF.load_state_dict(checkpoint['state_dict'])
    # baseF.eval()
    
    baseF = None
    
    F = GOTURNTracker(baseF)
    
    F.eval()
    
    F_cal = RegForecaster(F)
    
    return F_cal

def load_forecaster_full():

    ##FIXME: merge GoNet_var and GOTURNTracker
    baseF = GoNet_var()
    ##FIXME
    checkpoint = tc.load("otb/pytorch_goturn.pth.tar", map_location=lambda storage, loc: storage)
    #checkpoint = tc.load("pytorch_goturn.pth.tar", map_location=lambda storage, loc: storage) ##FIXME
    baseF.load_state_dict(checkpoint['state_dict'])
    baseF.eval()
    
    F = GOTURNTracker(baseF)
    
    F.eval()
    
    F_cal = RegForecaster(F)
    
    return F_cal


def xyxy2xywh(xyxy):
    xywh = xyxy.clone()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh

def xywh2xyxy(xywh):
    xyxy = xywh.clone()
    xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    return xyxy

def dist_mah(xs, cs, Ms, sqrt=True):
    diag = True if len(Ms.size()) == 2 else False
    assert(diag)
    assert(xs.size() == cs.size())
    assert(xs.size() == Ms.size())

    diff = xs - cs
    dist = diff.mul(Ms).mul(diff).sum(1)
    if sqrt:
        dist = dist.sqrt()
    return dist

class GOTURNTracker(Forecaster):
    def __init__(self, model, device=tc.device("cuda:0")):
        super().__init__()
        self.device = device

        self.model = model
        if model is not None:
            self.model = self.model.to(self.device)
                
    def encode_image(self, xs, ys, context=2, scaling=10, size=224):
        """
        encode an input image x with a bounding box y by including context, cropping the image, and resize the cropped image
        """
        xs_enc = []
        #ys_enc = []
        opts = []
        for x, y in zip(xs, ys):
            x_np = x.detach().cpu().numpy().transpose(1, 2, 0)            
            y_np = y.detach().cpu().numpy()

            # 1. crop the image incuding context
            sample = {}
            sample['image'] = x_np
            sample['bb'] = y_np
            x_enc, opt = crop_sample(sample) # internally define context=2

            # 2.resize the image
            samples = Rescale((size, size))(x_enc, opt)
            x_enc = transforms.ToTensor()(samples['image'])
            #y_enc = tc.tensor(samples['bb'])

            xs_enc.append(x_enc.unsqueeze(0))
            #ys_enc.append(y_enc.unsqueeze(0))
            opts.append(opt)
        xs_enc = tc.cat(xs_enc, 0)
        #ys_enc = tc.cat(ys_enc, 0)
        return xs_enc, opts

    def encode_bb(self, xs, ys, opts, context=2, scaling=10, size=224):
        img_curr = xs[2]
        bb_prev = xs[1]
        
        ys_enc = []
        for x, yh, y, opt in zip(img_curr, bb_prev, ys, opts):
            x_np = x.detach().cpu().numpy().transpose(1, 2, 0)            
            y_np = y.detach().cpu().numpy()
            yh_np = yh.detach().cpu().numpy()

            # 1. crop the image incuding context
            sample = {}
            sample['image'] = x_np
            sample['bb'] = y_np
            sample['bb_prev'] = yh_np
            samples, _ = crop_sample_bb(sample) # internally define context=2

            # 2.resize the image
            samples = Rescale((size, size))(samples, opt)
            y_enc = tc.tensor(samples['bb'])

            ys_enc.append(y_enc.unsqueeze(0))
        ys_enc = tc.cat(ys_enc, 0).to(ys.device)
        return ys_enc

    def decode_output(self, xs, yhs_feat, opts, context=2, scaling=10, size=224):
        """
        decode the output of neural network to get a bounding box in the image space.
        """

        img_curr = xs[2]
        
        yhs = []
        for yh_feat, x, opt in zip(yhs_feat, img_curr, opts):
            x_np = x.detach().cpu().numpy().transpose(1, 2, 0)
            yh_feat = yh_feat.detach().cpu().numpy()
            
            # 1. undo scalings
            bbox = BoundingBox(*yh_feat)
            bbox.unscale(opt['search_region'])
            bbox.uncenter(x_np, opt['search_location'], opt['edge_spacing_x'], opt['edge_spacing_y'])
            yh = tc.tensor([bbox.get_bb_list()])
            yhs.append(yh)
            
        yhs = tc.cat(yhs, 0)
        return yhs

    def forward(self, xs):
        """
        """
        if self.model is None:
            return xs

        # unpack inputs
        xs_prev, yhs_prev, xs_curr, *_ = xs
        
        # encode inputs
        xhs_prev, _ = self.encode_image(xs_prev, yhs_prev, 2, 10, 224)
        xhs, self.opts = self.encode_image(xs_curr, yhs_prev, 2, 10, 224)
        
        # predict a bb
        xhs_prev = xhs_prev.to(self.device)
        xhs = xhs.to(self.device)        
        yh_mu, yh_var = self.model((xhs_prev, xhs))
        
        return yh_mu, yh_var


class GoNet_var(Forecaster):
    """ Neural Network class
        Two stream model:
        ________
       |        | conv layers              Untrained Fully
       |Previous|------------------>|      Connected Layers
       | frame  |                   |    ___     ___     ___
       |________|                   |   |   |   |   |   |   |   fc4
                   Pretrained       |   |   |   |   |   |   |    * (left)
                   CaffeNet         |-->|fc1|-->|fc2|-->|fc3|--> * (top)
                   Convolution      |   |   |   |   |   |   |    * (right)
                   layers           |   |___|   |___|   |___|    * (bottom)
        ________                    |   (4096)  (4096)  (4096)  (4)
       |        |                   |
       | Current|------------------>|
       | frame  |
       |________|

    """
    def __init__(self):
        super(GoNet_var, self).__init__()
        caffenet = models.alexnet(pretrained=True)
        self.convnet = nn.Sequential(*list(caffenet.children())[:-1])
        for param in self.convnet.parameters():
            param.requires_grad = False
        self.regressor_mu = nn.Sequential(
                nn.Linear(256*6*6*2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4),
                )
        self.regressor_var = nn.Sequential(
                nn.Linear(256*6*6*2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4),
                )
        self.weight_init()

    def weight_init(self):
        for m in self.regressor_mu.modules():
            # fully connected layers are weight initialized with
            # mean=0 and std=0.005 (in tracker.prototxt) and
            # biases are set to 1
            # tracker.prototxt link: https://goo.gl/iHGKT5
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)
                

        for m in self.regressor_var.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    def normalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = tc.tensor(mean, device=x.device).unsqueeze(1).unsqueeze(2)
        std = tc.tensor(std, device=x.device).unsqueeze(1).unsqueeze(2)
        return (x - mean) / std

    def forward_mu(self, x, y):
        # normalize an image
        x = self.normalize(x)
        y = self.normalize(y)

        #print("%.11f, %.11f"%(x.mean(), y.mean()))
        
        x1 = self.convnet(x)
        x1 = x1.view(x.size(0), 256*6*6)
        x2 = self.convnet(y)
        x2 = x2.view(x.size(0), 256*6*6)
        x = tc.cat((x1, x2), 1)
        x = self.regressor_mu(x)
        return x

    def forward_var(self, x, y):
        # normalize an image
        x = self.normalize(x)
        y = self.normalize(y)
        
        x1 = self.convnet(x)
        x1 = x1.view(x.size(0), 256*6*6)
        x2 = self.convnet(y)
        x2 = x2.view(x.size(0), 256*6*6)
        x = tc.cat((x1, x2), 1)
        x = self.regressor_var(x)
        x = (x.exp() + 1).log()
        return x
    
    def forward(self, input):
        x, y = input[0], input[1]
        mus = self.forward_mu(x, y)
        vars = self.forward_var(x, y)
        return mus, vars

    def train_parameters(self):
        return [p for p in self.regressor_mu.parameters()] + [p for p in self.regressor_var.parameters()]


class RegForecaster(Forecaster):
    def __init__(self, baseF):
        super().__init__()
        self.baseF = baseF
        self.T = nn.Parameter(tc.ones(1))
        
    def forward(self, input):
        yhs, yhs_var = self.baseF(input)
        mu = yhs
        var = yhs_var / self.T
        return mu, var

    def neg_log_prob(self, yhs, yhs_var, ys):

        d = ys.size(1)

        if not all(yhs_var.view(-1) > 0):
            print(yhs_var)
            sys.exit()
        
        loss_mah = 0.5 * dist_mah(ys, yhs, 1/yhs_var, sqrt=False)

        assert(all(loss_mah >= 0))
        loss_const = 0.5 * T(2.0 * np.pi).log() * d
        loss_logdet = 0.5 * yhs_var.log().sum(1)
        loss = loss_mah + loss_logdet + loss_const

        return loss
            
    def cal_parameters(self):
        return [self.T]

    def train(self, mode=True):
        self.training = True
        self.baseF.eval()
