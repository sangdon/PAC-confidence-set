mkdir -p mpg/snapshots
CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task reg --dataset.name mpg --batch_size 10 --train.lr 0.0005 --cal.lr 1e-2 --cs.n 70 --cs.eps 0.1 --cs.delta 0.05 --train_cs > mpg/snapshots/output_est_conf_set_ours
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task reg --cs.no_cal --dataset.name mpg --batch_size 10 --cs.n 70 --cs.eps 0.1 --cs.delta 0.05 --train_cs > mpg/snapshots/output_est_conf_set_no_cal"
