import os, sys

sys.path.append("../../")
from calibration.TempReg import TempScalingReg as ForecasterLearner

##
## calibration for classification
##        
def cal_forecaster(params, F, ld_train, ld_val, no_cal):

    ## init
    fl = ForecasterLearner(params, F)
    
    ## test before cal
    fl.test([ld_val], ["val (before cal)"], model=F)

    ## calibrate
    if not no_cal:
        fl.train(ld_train, ld_val) 

    ## test after cal
    fl.test([ld_val], ["val (after cal)"], model=F)
        
    F.eval()
    return F



