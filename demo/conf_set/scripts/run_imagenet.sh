mkdir -p imagenet/snapshots
CUDA_VISIBLE_DEVICES=1 python3 pac_conf_set.py --task cls --train_cs --dataset.name imagenet --cs.n 20000 --cs.eps 0.01 0.02 0.03 0.04 0.05 \
--cs.delta 1e-1 1e-3 1e-5 > imagenet/snapshots/output_pac_conf_set_ours
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task cls --train_cs --dataset.name imagenet --cs.n 20000 --cs.eps 0.01 0.02 0.03 0.04 0.05 \
--cs.delta 1e-1 1e-3 1e-5 --cs.no_cal > imagenet/snapshots/output_pac_conf_set_no_cal"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task cls --train_cs --dataset.name imagenet --cs.n 20000 --cs.eps 0.01 0.02 0.03 0.04 0.05 \
--cs.delta 1e-1 1e-3 1e-5 --cs.no_db > imagenet/snapshots/output_pac_conf_set_no_db"
