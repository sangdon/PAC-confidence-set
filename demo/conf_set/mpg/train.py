import os, sys

sys.path.append("../../../")
from regression import learner

def train_forecaster(params, F, ld_train, ld_val, params_all=None):

    ## train
    sgd = learner.Regressor(params, F)
    
    sgd.set_opt_params(F.train_parameters())
    sgd.train(ld_train, ld_val)
    
    ## eval
    sgd.test([ld_val], ["val"])
    print()        
        
    return F                 

