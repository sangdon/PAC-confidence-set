# PAC Confidence Sets Construction
Implementation of PAC Confidence Sets for Deep Neural Network (ICLR20). This repository provides code to reproduce results of the paper on the three datasets (i.e., imagenet, mpg, and halfcheetah) for three tasks (i.e., classification, regression, and multi-dimensional regression).

Check the paper for details: https://arxiv.org/abs/2001.00106.

# Datasets Initialization

To initialize datasets, excute the following shell scripts for each dataset; each script download a dataset, extract it under `<repository root>/demo/conf_set/datasets`.

imagenet dataset:
```bash
cd <repository root>
./data/setup/init_imagenet.sh
```

mpg dataset:
```bash
cd <repository root>
./data/setup/init_mpg.sh
```

halfcheetah dataset:
```bash
cd <repository root>
./data/setup/init_halfcheetah.sh
```

# Usage

The following includes how to construct a confidence set predictor and how to plot results on confidence set size. The results are saved under `<repository root>/demo/conf_set/<dataset name>/snapshots/pac_conf_set`. For more usage examples, see `<repository root>/demo/conf_set/scripts`.

## construct a confidence set predictor

To construct a confidence set predictor, execute the following command for each dataset: 

imagenet dataset:
```bash
cd <repository root>/demo/conf_set
python3 pac_conf_set.py --task cls --dataset.name imagenet --cs.n 20000 --cs.eps 0.01 0.02 0.03 0.04 0.05 --cs.delta 1e-5 1e-3 1e-1 --train_cs
```

mpg dataset:
```bach
python3 pac_conf_set.py --task reg --dataset.name mpg --batch_size 10 --train.lr 0.0005 --cal.lr 1e-2 --cs.n 70 --cs.eps 0.1 0.2 --cs.delta 0.05 0.1 --train_cs
```

halfcheetah dataset:
```bash
python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.eps 0.01 0.02 0.03 0.04 0.05 --cs.delta 1e-5 1e-3 1e-1 --train_cs
```


## plot results on confidence set size

To draw the sensitivity analysis results of confidence set size over `\epsilon` and `\delta`, excute the following:

imagenet dataset:
```bash
cd <repository root>/demo/conf_set
python3 pac_conf_set.py --task cls --dataset.name imagenet --cs.n 20000 --cs.delta 1e-5 --cs.eps 0.01 0.02 0.03 0.04 0.05 --plot.eps
python3 pac_conf_set.py --task cls --dataset.name imagenet --cs.n 20000 --cs.eps 0.01 --cs.delta 1e-5 1e-3 1e-1 --plot.delta
```


mpg dataset:
```bash
cd <repository root>/demo/conf_set
python3 pac_conf_set.py --task reg --dataset.name mpg --cs.n 70 --cs.eps 0.1 0.2 --cs.delta 0.05 --plot.ylim 20.0 50.0 --plot.eps
python3 pac_conf_set.py --task reg --dataset.name mpg --cs.n 70 --cs.eps 0.1 --cs.delta 0.05 0.1 --plot.ylim 40.0 50.0 --plot.delta
```

halfcheetah dataset:
```bash
cd <repository root>/demo/conf_set
python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.delta 1e-5 --cs.eps 0.01 0.02 0.03 0.04 0.05 --plot.eps --plot.log_scale --plot.ylim 1.0 1e18
python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.eps 0.01 --cs.delta 1e-5 1e-3 1e-1 --plot.delta --plot.log_scale --plot.ylim 1.0 1e18
```




