screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --plot.traj --plot.log_scale --cs.n 5000"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=1 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --plot.box --plot.log_scale --cs.n 5000 --plot.ylim 1.0 1.5e25"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --plot.eps --plot.log_scale --cs.n 5000 --cs.delta 1e-5 --cs.eps 0.01 0.02 0.03 0.04 0.05 --plot.ylim 1.0 1e18"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --plot.delta --plot.log_scale --cs.n 5000 --cs.eps 0.01 --cs.delta 1e-5 1e-3 1e-1 --plot.ylim 1.0 1e18"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --plot.comp --cs.n 5000 --cs.eps 0.01 --cs.delta 1e-5 --plot.log_scale --plot.ylim 1.0 1e26"

