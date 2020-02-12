from torchvision import datasets, transforms
import torch as tc
import torch.utils.data as data
import sys, os
import types
import pickle
import shutil
import tempfile
import tarfile
import numpy as np
import time

from torchvision.datasets.utils import check_integrity, download_url

ARCHIVE_DICT = {
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}

def download_meta_file(root):
    meta_file = os.path.join(root, 'meta.bin')
    if not os.path.exists(meta_file):
        print("no longer supported")

def _load_meta_file(meta_file):
    if check_integrity(meta_file):
        return tc.load(meta_file)
    else:
        raise RuntimeError("Meta file not found or corrupted.",
                           "You can use download=True to create it.")
        
def label_to_name(root, label_to_wnid):
    meta_file = os.path.join(root, 'meta.bin')
    if not os.path.exists(meta_file):
        return []
    wnid_to_names = _load_meta_file(meta_file)[0]

    names = [wnid_to_names[wnid][0].replace(' ', '_').replace('\'', '_') for wnid in label_to_wnid]
    return names
        
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

def init_ld(root_dir, tform, batch_size, shuffle, num_workers, load_type):

    if load_type == "none":
        if os.path.exists(root_dir):
            data = datasets.ImageFolder(root=root_dir, transform=tform)
        else:
            data = None
    elif "feature" in load_type or "logit" in load_type:
        def pickle_loader(input):
            return pickle.load(open(input, 'rb'))
        target_dir = root_dir+"_%s"%(load_type)
        if os.path.exists(target_dir):
            data = datasets.DatasetFolder(root=target_dir, loader=pickle_loader, extensions=("pk"))
        else:
            data = None
    if data is None:
        return None
    else:
        return tc.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, 
            drop_last=False, num_workers=num_workers)


def shuffle_initial_data_order(ld, seed):
    if ld is None:
        return
    
    n_data = len(ld.dataset.samples)
    np.random.seed(seed)
    idx_rnd = np.random.permutation(n_data)
    ld.dataset.samples = [ld.dataset.samples[i] for i in idx_rnd]
    ld.dataset.targets = [ld.dataset.targets[i] for i in idx_rnd]
    if hasattr(ld.dataset, "imgs"):
        ld.dataset.imgs = ld.dataset.samples
    np.random.seed(int(time.time()%2**32))    

def loadImageNet_CS(root_dir, batch_size,
                    train_shuffle=True, val1_train_shuffle=True, val1_shuffle=False, val2_shuffle=False, test_shuffle=False, 
                    post_fix="", load_type='none', num_workers=8, f_normalize=True):
    
    ld = types.SimpleNamespace()
    
    if f_normalize:
        tform_rnd = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        tform_no_rnd = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             normalize])
    else:
        tform_rnd = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        tform_no_rnd = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor()])
        
    if load_type == 'none':
        tform_tr = tform_rnd if train_shuffle else tform_no_rnd
        tform_val1_tr = tform_rnd if val1_train_shuffle else tform_no_rnd
        tform_val1 = tform_no_rnd
        tform_val2 = tform_no_rnd
        tform_te = tform_no_rnd
    else:
        tform_tr = None
        tform_val1_tr = None
        tform_val1 = None
        tform_val2 = None
        tform_te = None
    
    ### train
    ld.train = None

    # val1_tr
    root_dir_val1_tr = os.path.join(root_dir, 'val1'+post_fix)
    ld.val1_train = init_ld(root_dir_val1_tr, tform_val1_tr, batch_size, val1_train_shuffle, num_workers, load_type)
    shuffle_initial_data_order(ld.val1_train, 0)

    # val1
    root_dir_val1 = os.path.join(root_dir, 'val1'+post_fix)
    ld.val1 = init_ld(root_dir_val1, tform_val1, batch_size, val1_shuffle, num_workers, load_type)
    shuffle_initial_data_order(ld.val1, 1)
    
    # val2
    root_dir_val2 = os.path.join(root_dir, 'val2'+post_fix)
    ld.val2 = init_ld(root_dir_val2, tform_val2, batch_size, val2_shuffle, num_workers, load_type)
    shuffle_initial_data_order(ld.val2, 2)
    
    # test
    root_dir_te = os.path.join(root_dir, 'test'+post_fix)
    ld.test = init_ld(root_dir_te, tform_te, batch_size, test_shuffle, num_workers, load_type)
    shuffle_initial_data_order(ld.test, 3)
    
    # label, name map
    #download_meta_file(root_dir)
    if ld.test is not None:
        ld.names = label_to_name(root_dir, ld.test.dataset.classes)
    
    return ld


