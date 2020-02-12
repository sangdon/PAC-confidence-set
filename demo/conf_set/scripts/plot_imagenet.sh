screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 pac_conf_set.py --task cls --dataset.name imagenet --plot.box --cs.n 20000 --cs.eps 0.01 0.02 0.03 0.04 0.05 --cs.delta 1e-5 1e-3 1e-1"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=1 python3 pac_conf_set.py --task cls --dataset.name imagenet --plot.eps --cs.n 20000 --cs.delta 1e-5 --cs.eps 0.01 0.02 0.03 0.04 0.05"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task cls --dataset.name imagenet --plot.delta --cs.n 20000 --cs.eps 0.01 --cs.delta 1e-5 1e-3 1e-1"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task cls --dataset.name imagenet --plot.cond --cs.n 20000 --cs.eps 0.01 --cs.delta 1e-5"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 pac_conf_set.py --task cls --dataset.name imagenet --plot.comp --cs.n 20000 --cs.eps 0.01 --cs.delta 1e-5"