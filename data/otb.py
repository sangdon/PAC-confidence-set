from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch as tc
import torch.tensor as T

import os, sys
import glob
import types
import numpy as np
import random
import cv2
import time
from PIL import Image, ImageDraw
import pickle

def xywh2xyxy(xywh):
    xyxy = xywh.clone()
    if len(xyxy.size()) == 2:
        xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    else:
        xyxy[2:] = xywh[:2] + xywh[2:]
    return xyxy


def resize_input(img, bb_xywh, frame_wh):
    img_resized = cv2.resize(img, frame_wh)
    x_scale = float(frame_wh[0]) / float(img.shape[1])
    y_scale = float(frame_wh[1]) / float(img.shape[0])
    bb_resized = [
        bb_xywh[0] * x_scale,
        bb_xywh[1] * y_scale,
        bb_xywh[2] * x_scale,
        bb_xywh[3] * y_scale
    ]
    return img_resized, bb_resized

def plot_bb(img, bb_xywh, fn=None):
    img_PIL = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img_PIL)
    draw.rectangle((*bb_xywh[:2], *(bb_xywh[:2]+bb_xywh[2:])), outline="white", width=2)
    if fn is not None:
        img_PIL.save(fn)
    else:
        return img_PIL
    
class OTBDataset(Dataset):
    def __init__(self, root, load_type, frame_wh=(320, 240), img_type="float", bb_format="xywh"):
        self.load_type = load_type
        self.frame_wh = frame_wh
        self.img_type = img_type
        self.bb_format = bb_format
        
        # load seq ids
        self.seq_ids = os.listdir(root)
        
        # load images
        self.frames = sorted(glob.glob(os.path.join(root, "**/img/*.jpg")))

        # keep start frame_id
        self.start_frame_id = {}
        for seq_id in self.seq_ids:
            for f in self.frames:
                if seq_id == f.split("/")[-3]:
                    self.start_frame_id[seq_id] = int(f.split("/")[-1].split(".")[0])
                    break

        # keep a pair of two adjacent frames
        self.frames_pair = []
        for fn_prev, fn_curr in zip(self.frames[:-1], self.frames[1:]):
            seq_id_prev = fn_prev.split("/")[-3]
            seq_id_curr = fn_curr.split("/")[-3]
            if seq_id_prev != seq_id_curr:
                continue
            self.frames_pair.append([fn_prev, fn_curr])

        # load bounding boxes: assume single-object tracking
        self.bbox = {}
        for seq_id in self.seq_ids:
            root_seq = os.path.join(root, seq_id)
            bb_fns = glob.glob(os.path.join(root_seq, "*.txt"))
            try:
                self.bbox[seq_id] = T(np.loadtxt(bb_fns[-1], delimiter=","))
            except ValueError:
                self.bbox[seq_id] = T(np.loadtxt(bb_fns[-1], delimiter="\t"))
                
        # tform
        self.tform_img = transforms.Compose([
            transforms.ToTensor()
        ])
                
    def __len__(self):
        #if self.load_type == "tracking":
        #    return len(self.frames)
        #else:
        return len(self.frames_pair)
    
    def __getitem__(self, idx):
        # if self.load_type == "tracking":
        #     # read an image
        #     img = cv2.imread(self.frames[idx])
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #     # # read and transform an image
        #     # img = Image.open(self.frames[idx])
        #     # img = img.convert("RGB")
        #     #img = self.tform_img(img)
        #     if self.img_type == "float":
        #         img = transforms.ToTensor()(img)
        #     else:
        #         assert(self.img_type=="uint8")    
        #         img = tc.tensor(img.transpose(2, 0, 1))

        #     # read and transform a bb
        #     splits = self.frames[idx].split("/")        
        #     seq_id = splits[-3]
        #     frame_id = int(splits[-1].split(".")[0])
        #     start_frame_id = self.start_frame_id[seq_id]
        #     bb = self.bbox[seq_id][frame_id - start_frame_id -1] # (x, y, w, h)
        #     bb = T(bb)
        #     if self.bb_format == "xyxy":
        #         bb = xywh2xyxy(bb)
        #     #h, w = img.shape[:2]
        #     #bb = self.tform_bb(bb, h, w, self.size)

        #     return ([], [], img, seq_id, T(frame_id)), T(bb) 
        # else:
        
        # read previous and current images
        frame_prev = self.frames_pair[idx][0]
        frame_curr = self.frames_pair[idx][1]
        img_prev = cv2.imread(frame_prev)
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2RGB) if img_prev.ndim == 2 else cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB)
        img_curr = cv2.imread(frame_curr)
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2RGB) if img_curr.ndim == 2 else cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

        # read and transform a bb
        splits = frame_prev.split("/")
        seq_id = splits[-3]
        frame_id = int(splits[-1].split(".")[0])
        start_frame_id = self.start_frame_id[seq_id]
        bb_prev = self.bbox[seq_id][frame_id - start_frame_id -1] # (x, y, w, h)
        seq_id_prev = seq_id
        frame_id_prev = T(frame_id)

        splits = frame_curr.split("/")        
        seq_id = splits[-3]
        frame_id = int(splits[-1].split(".")[0])
        start_frame_id = self.start_frame_id[seq_id]
        bb_curr = self.bbox[seq_id][frame_id - start_frame_id -1] # (x, y, w, h)
        seq_id_curr = seq_id
        frame_id_curr = T(frame_id)

        # resize
        if self.load_type != "tracking":
            img_prev, bb_prev = resize_input(img_prev, bb_prev, self.frame_wh)
            img_curr, bb_curr = resize_input(img_curr, bb_curr, self.frame_wh)          

        # reformat
        if self.img_type == "float":
            img_prev = transforms.ToTensor()(img_prev)
            img_curr = transforms.ToTensor()(img_curr)
        else:
            assert(self.img_type=="uint8")    
            img_prev = T(img_prev.transpose(2, 0, 1))
            img_curr = T(img_curr.transpose(2, 0, 1))
                
        bb_prev = T(bb_prev)
        bb_curr = T(bb_curr)
        if self.bb_format == "xyxy":
            bb_prev = xywh2xyxy(bb_prev)
            bb_curr = xywh2xyxy(bb_curr)
            
        if self.load_type == "tracking":
            return (img_prev, bb_prev, seq_id_prev, frame_id_prev, img_curr, seq_id_curr, frame_id_curr), bb_curr
        else:
            return (img_prev, bb_prev, img_curr), bb_curr
        
class RegressionDataset(Dataset):
    def __init__(self, root_dir):
        self.fns = glob.glob(os.path.join(root_dir, "*.pk"))
        
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        with open(self.fns[idx], "rb") as f:
            yh, yh_var, y = pickle.load(f)
            return [yh, yh_var], y

def init_ld(root_dir, batch_size, shuffle, num_workers, load_type, img_type, bb_format, load_precomp=False):

    target_dir = root_dir
    assert(os.path.exists(target_dir))

    if load_precomp:
        assert(bb_format=="xyxy")
        ds = RegressionDataset(target_dir)
    else:
        ds = OTBDataset(target_dir, load_type, img_type=img_type, bb_format=bb_format)
                                
    return tc.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, 
        drop_last=False, num_workers=num_workers)

def shuffle_initial_data_order(ld, seed):
    if hasattr(ld.dataset, "frames_pair"):
        n_data = len(ld.dataset.frames_pair)
        random.seed(seed)
        random.shuffle(ld.dataset.frames_pair)
        random.seed(int(time.time()%2**32))
    else:
        n_data = len(ld.dataset.fns)
        random.seed(seed)
        random.shuffle(ld.dataset.fns)
        random.seed(int(time.time()%2**32))

def loadOTB(
        root_dir, batch_size, 
        train_shuffle=True,
        val1_train_shuffle=True,
        val1_shuffle=False,
        val2_shuffle=False,
        test_shuffle=False,
        num_workers=4,
        img_type="float",
        bb_format="xywh",
        load_precomp=False):

    ld = types.SimpleNamespace()

    # train
    ld.train = None

    # val1
    root_dir_val1 = os.path.join(root_dir, 'val1')
    ld.val1 = init_ld(root_dir_val1, batch_size, val1_shuffle, num_workers, load_type="val1", img_type=img_type, bb_format=bb_format, load_precomp=load_precomp)
    shuffle_initial_data_order(ld.val1, 0)
        
    # val1_train
    root_dir_val1_tr = os.path.join(root_dir, 'val1')
    ld.val1_train = init_ld(root_dir_val1_tr, batch_size, val1_train_shuffle, num_workers, load_type="val1_train", img_type=img_type, bb_format=bb_format, load_precomp=load_precomp)
    shuffle_initial_data_order(ld.val1_train, 0)

    # val2
    root_dir_val2 = os.path.join(root_dir, 'val2')
    ld.val2 = init_ld(root_dir_val2, batch_size, val2_shuffle, num_workers, load_type="val2", img_type=img_type, bb_format=bb_format, load_precomp=load_precomp)
    shuffle_initial_data_order(ld.val2, 0)

    # test
    root_dir_test = os.path.join(root_dir, 'test')
    ld.test = init_ld(root_dir_test, batch_size, test_shuffle, num_workers, load_type="test", img_type=img_type, bb_format=bb_format, load_precomp=load_precomp)
    shuffle_initial_data_order(ld.test, 0)
    
    # tracking
    if not load_precomp:
        root_dir_tracking = os.path.join(root_dir, 'test')
        ld.tracking = init_ld(root_dir_tracking,  1, False, 0, load_type="tracking", img_type=img_type, bb_format=bb_format) # always batch_size = 1, num_workers=0

    return ld
    
if __name__ == "__main__":

    ld = loadOTB("/home/sangdonp/Research/datasets/OTB", 100)

    # n_total = 0.0
    # for xs, ys in ld.val1:
    #     n_total += ys.size(0)
    # print("# n_val1 = %d"%(n_total))

    root_res = "plot_bb/val1"
    os.makedirs(root_res, exist_ok=True)
    i = 0
    for xs, ys in ld.val1:
        for x, y in zip(xs[0], ys):
            fn = os.path.join(root_res, "%.4d.png"%i)
            plot_bb(x, y, fn)
            i += 1
    sys.exit()
    
    # n_total = 0.0
    # for xs, ys in ld.val1_train:
    #     n_total += ys.size(0)
    # print("# n_val1_train = %d"%(n_total))

    n_total = 0.0
    for xs, ys in ld.val2:
        n_total += ys.size(0)
    print("# n_val2 = %d"%(n_total))

    n_total = 0.0
    for xs, ys in ld.test:
        n_total += ys.size(0)
    print("# n_test = %d"%(n_total))


"""
# n_val1 = 20882
# n_val2 = 22761
# n_test = 14297
"""
