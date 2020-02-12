import argparse
import os, sys
import numpy as np
import torch as tc

sys.path.append("../../")

def to_tree_namespace(params):
    """
    handle only depth-1 tree
    """
    for key, val in vars(params).copy().items():
        if "." in key:
            delattr(params, key)
            group, sub_key = key.split(".", 2)
            if not hasattr(params, group):
                setattr(params, group, argparse.Namespace())
            setattr(getattr(params, group), sub_key, val)
    return params
            
def print_params(params, param_str="parameters", n_tap_str=1):
    print("\t"*(n_tap_str-1)+param_str + ":" )
    for key, val in vars(params).items():
        if "Namespace" in str(type(val)):
            print_params(val, param_str=key, n_tap_str=n_tap_str+1)
        else:
            print("\t"*n_tap_str + key + ":", val)
    print()
    
class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
            fromfile_prefix_chars='@')

        ## meta options  
        self.parser.add_argument('--exp_name', type=str, required=False, 
                                 help='name of an experiment')
        self.parser.add_argument('--snapshot_root', type=str, required=False, 
                                 help='snapshot root')
        self.parser.add_argument('--use_cpu', action="store_true", 
                                 help='use CPU')
        self.parser.add_argument('--task', type=str, required=True, 
                                 help='classification (cls), regression (reg), or RL (rl) tasks')
        self.parser.add_argument('--batch_size', type=int, default=100, 
                                 help='the batch size of a data loader')
        self.parser.add_argument('--train_cs', action="store_true", 
                                 help='train (and test) or plot')

        ## architecture
        self.parser.add_argument('--arch.name', type=str, required=False,
                                 help='architecture name (currently only for classification using resnet152)')
        
        ## dataset
        self.parser.add_argument('--dataset.name', type=str, required=True, 
                                 help='dataset name [imagenet | mpg | halfcheetah]')

        self.parser.add_argument('--dataset.load_type', type=str, required=False, 
                                 help='dataset load type [ ResNet152_logits]')
        self.parser.add_argument('--dataset.num_workers', type=int, default=4, 
                                 help='the number of workders for the pytorch dataloader')

        ## train parameters
        self.parser.add_argument('--train.optimizer', type=str, required=False, default='SGD',
                                 help='name of an experiment')
        self.parser.add_argument('--train.n_epochs', type=int, default=5000, 
                                 help='the number of epochs for calibration')
        self.parser.add_argument('--train.lr', type=float, default=1e-2, 
                                 help='learning rate for calibration')
        self.parser.add_argument('--train.lr_decay_epoch', type=int, default=1000, 
                                 help='lr decay epochs for calibration')
        self.parser.add_argument('--train.lr_decay_rate', type=float, default=0.5, 
                                 help='lr decay rate for calibration')
        self.parser.add_argument('--train.momentum', type=float, default=0.9, 
                                 help='SGD momentum for calibration')
        self.parser.add_argument('--train.weight_decay_sgd', type=float, default=0.0, 
                                 help='SGD weight decay for calibration')
        self.parser.add_argument('--train.label_weight', type=float, nargs='*',
                                 help='a list of label weight parameters')
        
        
        ## calibration parameters
        self.parser.add_argument('--cal.test', action="store_true", 
                                 help='enable test for calibrated models')
        self.parser.add_argument('--cal.optimizer', type=str, required=False, default='SGD',
                                 help='name of an experiment')
        self.parser.add_argument('--cal.n_epochs', type=int, default=1000, 
                                 help='the number of epochs for calibration')
        self.parser.add_argument('--cal.lr', type=float, default=1e-2, 
                                 help='learning rate for calibration')
        self.parser.add_argument('--cal.lr_decay_epoch', type=int, default=200, 
                                 help='lr decay epochs for calibration')
        self.parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5, 
                                 help='lr decay rate for calibration')
        self.parser.add_argument('--cal.momentum', type=float, default=0.9, 
                                 help='SGD momentum for calibration')
        self.parser.add_argument('--cal.weight_decay_sgd', type=float, default=0.0, 
                                 help='SGD weight decay for calibration')
        self.parser.add_argument('--cal.label_weight', type=float, nargs='*',
                                 help='a list of label weight parameters')
        self.parser.add_argument('--cal.T', type=float, 
                                 help='manual calibration parameter')
    
        ## confidence set specific parameters
        self.parser.add_argument('--cs.no_acc', action="store_true", 
                                 help='not use accumulation')
        self.parser.add_argument('--cs.no_cal', action="store_true", 
                                 help='not use calibration')
        self.parser.add_argument('--cs.no_db', action="store_true", 
                                 help='not use direct bound')
        self.parser.add_argument('--cs.T', type=float, 
                                 help='heuristic confidence set level')
        self.parser.add_argument('--cs.T_min', type=float, 
                                 help='minimum search range of T')
        self.parser.add_argument('--cs.T_max', type=float, 
                                 help='maximum search range of T')
        self.parser.add_argument('--cs.T_diff', type=float, 
                                 help='robust parameter')

        self.parser.add_argument('--cs.no_error', action="store_true", 
                                 help='do not allow errors (i.e., k=0)')
        self.parser.add_argument('--cs.cond_thres', action="store_true", 
                                 help='baseline')
        self.parser.add_argument('--cs.log_scale', action="store_true", 
                                 help='compute log-scaled confidence set size')

        ## confidence set user parameters
        self.parser.add_argument('--cs.n', type=int, nargs='*', default=[20000, 15000, 10000, 5000], 
                                 help='a list of n')
        self.parser.add_argument('--cs.eps', type=float, nargs='*', default=[0.01, 0.02, 0.03, 0.04, 0.05], 
                                 help='a list of epsilon')
        self.parser.add_argument('--cs.delta', type=float, nargs='*', default=[1e-5, 1e-3, 1e-1], 
                                 help='a list of delta')

        ## plots
        self.parser.add_argument('--plot.comp', action="store_true", 
                                 help='plot comparison results')
        self.parser.add_argument('--plot.box', action="store_true", 
                                 help='plot box plots')
        self.parser.add_argument('--plot.traj', action="store_true", 
                                 help='plot trajectories')
        self.parser.add_argument('--plot.delta', action="store_true", 
                                 help='plot a delta-dependecy plot')
        self.parser.add_argument('--plot.eps', action="store_true", 
                                 help='plot a epsilon-dependecy plot')
        self.parser.add_argument('--plot.cond', action="store_true", 
                                 help='plot a conditional conf set size plot')
        self.parser.add_argument('--plot.exs', action="store_true", 
                                 help='plot examples')
        self.parser.add_argument('--plot.confexs', action="store_true", 
                                 help='plot confident examples')


        ## for trajectory plot
        self.parser.add_argument('--plot.ex_summary', type=str, default='none', 
                                 help='a type to summary trajectories')
        ## for all plot
        self.parser.add_argument('--plot.time_summary', type=str, 
                                 help='a type to summary an example') ##TODO: check possible values
        self.parser.add_argument('--plot.log_scale', action="store_true", 
                                 help='plot log-scale y axis')        
        self.parser.add_argument('--plot.ylim', type=float, nargs='*', 
                                 help='a list of delta')
        self.parser.add_argument('--plot.font_size', type=int, default=20, 
                                 help='figure font size')


    def write_imagenet_default(self, params):

        assert(params.task == 'cls')
        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "imagenet/snapshots"
            if params.arch.name is not None:
                params.snapshot_root += "_" + params.arch.name
        
        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        if params.dataset.load_type is None:
            if params.arch.name is None:
                params.dataset.load_type = "ResNet152_logits"
            elif params.arch.name.lower() == "resnet152":
                params.dataset.load_type = "ResNet152_logits"
            elif params.arch.name.lower() == "alexnet":
                params.dataset.load_type = "AlexNet_logits"
            elif params.arch.name.lower() == "googlenet":
                params.dataset.load_type = "GoogLeNet_logits"            
            elif params.arch.name.lower() == "vgg19":
                params.dataset.load_type = "VGG19_logits"
            else:
                raise NotImplementedError            
        params.dataset.n_labels = 1000

        from data.imagenet import loadImageNet_CS as loadImageNet
        params.dataset.ld = loadImageNet(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers, load_type=params.dataset.load_type,
            val2_shuffle=False, test_shuffle=False)
        params.dataset.ld_ori = loadImageNet(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers, load_type="none",
            val2_shuffle=False, test_shuffle=False, f_normalize=False)
        
        ## forecaster train/calibration
        from imagenet.forecasters import load_forecaster
        from imagenet.train import train_forecaster
        from imagenet.calibrate import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster
        
        return params

    def write_alarm_default(self, params):
        ## binary classification on the alarm dataset
        assert(params.task == "cls")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "alarm/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        params.dataset.n_labels = 2
        
        from data.alarm import loadAlarm
        params.dataset.ld = loadAlarm(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from alarm.forecasters import load_forecaster
        from alarm.train import train_forecaster
        from alarm.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    def write_heart_default(self, params):
        ## binary classification on the heart dataset
        assert(params.task == "cls")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "heart/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        params.dataset.n_labels = 5
        
        from data.heart import loadHeart
        params.dataset.ld = loadHeart(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from heart.forecasters import load_forecaster
        from heart.train import train_forecaster
        from heart.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    def write_arrhythmia_default(self, params):
        ## binary classification on the arrhythmia dataset
        assert(params.task == "cls")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "arrhythmia/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        params.dataset.n_labels = 2 # presence and no presence
        
        from data.arrhythmia import loadArrhythmia
        params.dataset.ld = loadArrhythmia(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from arrhythmia.forecasters import load_forecaster
        from arrhythmia.train import train_forecaster
        from arrhythmia.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    def write_wine_default(self, params):
        ## binary classification on the wine dataset
        assert(params.task == "cls")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "wine/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        params.dataset.n_labels = 3
        
        from data.wine import loadWine as loader
        params.dataset.ld = loader(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from wine.forecasters import load_forecaster
        from wine.train import train_forecaster
        from wine.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    def write_car_default(self, params):
        ## multiclass classification on the car evaluation xsdataset
        assert(params.task == "cls")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "car/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        params.dataset.n_labels = 4
        
        from data.car import loadCar as loader
        params.dataset.ld = loader(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from car.forecasters import load_forecaster
        from car.train import train_forecaster
        from car.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    
    def write_halfcheetah_default(self, params):

        assert(params.task == 'rl')
        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "halfcheetah/snapshots"
        
        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        from data.halfcheetah import loadHalfCheetah
        params.dataset.ld = loadHalfCheetah(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        params.dataset.n_steps = 20
        params.dataset.n_states = 18
        
        ## forecaster train/calibration
        from halfcheetah.forecasters import load_forecaster
        from halfcheetah.train import train_forecaster
        from halfcheetah.calibrator import cal_forecaster        
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster
                
        return params

    def write_mpg_default(self, params):
        ## regression
        assert(params.task == "reg")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "mpg/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        
        from data.mpg import loadMPG as loader
        params.dataset.ld = loader(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from mpg.forecasters import load_forecaster
        from mpg.train import train_forecaster
        from mpg.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params
    
    def write_students_default(self, params):
        ## regression
        assert(params.task == "reg")

        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "students/snapshots"

        ## dataset
        params.dataset.path = os.path.join("datasets", params.dataset.name)
        
        from data.students import loadSTUDENTS as loader
        params.dataset.ld = loader(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False)
        
        ## forecaster train/calibration
        from students.forecasters import load_forecaster
        from students.train import train_forecaster
        from students.calibrator import cal_forecaster
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster

        return params

    
    def write_otb_default(self, params):

        assert(params.task == 'reg')
        ## snapshot_root
        if params.snapshot_root is None:
            params.snapshot_root = "otb/snapshots" 
        
        ## dataset
        #params.dataset.precomp = True
        dataset_postfix = "_precomp"
        params.dataset.path = os.path.join("datasets", params.dataset.name + dataset_postfix)
        from data.otb import loadOTB
        params.dataset.ld = loadOTB(
            params.dataset.path, params.batch_size,
            num_workers=params.dataset.num_workers,
            val2_shuffle=False, test_shuffle=False, bb_format="xyxy", load_precomp=True)
        
        ## forecaster train/calibration
        from otb.forecasters import load_forecaster
        from otb.train import train_forecaster
        from otb.calibrator import cal_forecaster                
        params.load_forecaster = load_forecaster
        params.train_forecaster = train_forecaster
        params.cal_forecaster = cal_forecaster
        
        return params

    def read_args(self, exp_name=None):
        ## read options
        params, _ = self.parser.parse_known_args()
        params = to_tree_namespace(params)

        ## check parameters are valid
        assert(params.task in ['cls', 'reg', 'rl'])
        
        ## fill default parameters
        if params.exp_name is None:
            params.exp_name = exp_name            
        params.use_gpu = not params.use_cpu

            
        ## fill dataset-dependent default parameters
        if "imagenet" in params.dataset.name.lower():
            params = self.write_imagenet_default(params)
        elif "alarm" in params.dataset.name.lower():
            params = self.write_alarm_default(params)
        elif "heart" in params.dataset.name.lower():
            params = self.write_heart_default(params)
        elif "arrhythmia" in params.dataset.name.lower():
            params = self.write_arrhythmia_default(params)
        elif "wine" in params.dataset.name.lower():
            params = self.write_wine_default(params)
        elif "car" in params.dataset.name.lower():
            params = self.write_car_default(params)            
        elif "mpg" in params.dataset.name.lower():
            params = self.write_mpg_default(params)
        elif "students" in params.dataset.name.lower():
            params = self.write_students_default(params)
        elif "otb" in params.dataset.name.lower():
            params = self.write_otb_default(params)
        elif "halfcheetah" in params.dataset.name.lower():
            params = self.write_halfcheetah_default(params)
        else:
            raise NotImplementedError

        ## load conf_set predictors
        if params.task == "cls":
            from conf_set.conf_set import ConfSetCls as ConfSetModel
            from conf_set.conf_set import ConfSetCls_cond as ConfSetModel_cond
        else:
            from conf_set.conf_set import ConfSetReg as ConfSetModel
            from conf_set.conf_set import ConfSetReg_cond as ConfSetModel_cond
        params.ConfSet = ConfSetModel
        params.ConfSet_cond = ConfSetModel_cond 


        ## fill train/cal parameters
        params.train.snapshot_root = params.snapshot_root
        params.train.load_model = True
        params.train.save_model = True
        params.train.use_gpu = params.use_gpu
        params.train.keep_best = True
        
        params.train.n_epoch_eval = np.inf
        params.train.early_term_cri = 0.2

        params.cal.snapshot_root = params.snapshot_root
        params.cal.load_model = True
        params.cal.save_model = True
        params.cal.use_gpu = params.use_gpu
        params.cal.keep_best = True
        params.cal.n_epoch_eval = np.inf
        params.cal.early_term_cri = 0.2

        ## fill cs default parameters
        params.cs.load_model = True
        params.cs.save_model = True
        if params.task == "cls":
            if params.cs.T_min is None:
                params.cs.T_min = 0.0
            if params.cs.T_max is None:
                params.cs.T_max = 1.0
            if params.cs.T_diff is None:
                params.cs.T_diff = 1e-16
        else:
            if params.cs.T_min is None:
                params.cs.T_min = -1e16
            if params.cs.T_max is None:
                params.cs.T_max = 1e16
            if params.cs.T_diff is None:
                params.cs.T_diff = 1e-6
        
        # ## confidence set
        # if params.cs.no_pac:
        #     assert(params.cs.T is not None)
                    
        ## plots
        if params.plot.ylim is None:
            if params.task == "cls":
                params.plot.ylim = [0.0, float(params.dataset.n_labels)]
            elif params.task == "reg":
                params.plot.ylim = [0.0, 20.0]
            elif params.task == "rl":
                if params.plot.log_scale:
                    params.plot.ylim = [1, 1e28]
                else:
                    params.plot.ylim = [0.0, 1.0]
            else:
                raise NotImplementedError
        params.plot.broken = False
        params.plot.symlog_scale = False

        ## plot.ex_summary
        if params.plot.traj:
            params.plot.ex_summary = 'mean'
            
        ## plot.time_summary            
        if params.task == "rl":
            if params.plot.time_summary is None:
                params.plot.time_summary = 'none' if params.plot.traj else 'all'
                
            if params.plot.time_summary == 'none':
                params.plot.time_summary = []
            elif params.plot.time_summary == 'all':
                params.plot.time_summary = [-1]
            else:
                step_idx = int(params.plot.time_summary)
                assert(step_idx >=0 and step_idx < params.dataset.n_steps)
                params.plot.time_summary = [step_idx]

        ## type conversion
        if params.train.label_weight is not None:
            params.train.label_weight = tc.tensor(params.train.label_weight)
            if params.use_gpu:
                params.train.label_weight = params.train.label_weight.cuda()
        if params.cal.label_weight is not None:
            params.cal.label_weight = tc.tensor(params.cal.label_weight)
            if params.use_gpu:
                params.cal.label_weight = params.cal.label_weight.cuda()
                
                
        ## return
        print_params(params)
        return params
