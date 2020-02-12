from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch as tc
import torch.tensor as T

import os, sys
import pickle
import glob
import types

class RegressionDataset(Dataset):
    def __init__(self, root_dir):
        self.fns = glob.glob(os.path.join(root_dir, "*.pk"))
        
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        with open(self.fns[idx], "rb") as f:
            xs, ys, yhs, yhs_var = pickle.load(f)
            return (T(xs, dtype=tc.float), T(yhs, dtype=tc.float), T(yhs_var, dtype=tc.float)), T(ys, dtype=tc.float)
        
def init_ld(root_dir, batch_size, shuffle, num_workers):
    
    target_dir = root_dir
    print(target_dir)
    assert(os.path.exists(target_dir))
    data = RegressionDataset(target_dir)
    
    return tc.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, 
        drop_last=False, num_workers=num_workers)

def loadHalfCheetah(
        root_dir, batch_size, 
        train_shuffle=True,
        val1_train_shuffle=True,
        val1_shuffle=False,
        val2_shuffle=False, test_shuffle=False, num_workers=4):

    ld = types.SimpleNamespace()

    # train
    ld.train = None
    
    # val1
    root_dir_val1 = os.path.join(root_dir, 'val1')
    ld.val1 = init_ld(root_dir_val1, batch_size, val1_shuffle, num_workers)
    
    # val1_train
    root_dir_val1_tr = os.path.join(root_dir, 'val1')
    ld.val1_train = init_ld(root_dir_val1_tr, batch_size, val1_train_shuffle, num_workers)

    # val
    root_dir_val2 = os.path.join(root_dir, 'val2')
    ld.val2 = init_ld(root_dir_val2, batch_size, val2_shuffle, num_workers)
    
    # test
    root_dir_te = os.path.join(root_dir, 'test')
    ld.test = init_ld(root_dir_te,  batch_size, test_shuffle, num_workers)

    return ld
    
if __name__ == "__main__":

    ld = loadHalfCheetah("datasets/halfcheetah", 100)
    
    n_total = 0.0
    for xs, ys in ld.val:
        n_total += ys.size(0)
    print("# n_val = %d"%(n_total))

    n_total = 0.0
    for xs, ys in ld.test:
        n_total += ys.size(0)
    print("# n_test = %d"%(n_total))

