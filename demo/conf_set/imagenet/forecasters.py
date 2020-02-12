import os, sys

def load_forecaster(params):

    if (params.arch.name is None) or (params.arch.name.lower() == "resnet152"):
        from models.ResNet import ResNet152 as Model
    elif params.arch.name.lower() == "googlenet":
        from models.CNN import GoogLeNet as Model
    elif params.arch.name.lower() == "alexnet":
        from models.CNN import AlexNet as Model
    elif params.arch.name.lower() == "vgg19":
        from models.CNN import VGG19 as Model
    else:
        raise NotImplementedError

    from models.ScalarForecasters import TempForecaster as CalForecaster

    F = Model(load_type=params.dataset.load_type)
    F_cal = CalForecaster(F)

    return F_cal


