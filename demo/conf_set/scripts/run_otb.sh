mkdir -p otb/snapshots
screen -dm bash -c "CUDA_VISIBLE_DEVICES=1 python3 pac_conf_set.py --task reg --dataset.name otb --train_cs > otb/snapshots/output_est_conf_set_ours"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task reg --dataset.name otb --cs.no_cal --train_cs > otb/snapshots/output_est_conf_set_no_cal"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 pac_conf_set.py --task reg --dataset.name otb --cs.no_db --train_cs > otb/snapshots/output_est_conf_set_no_db"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task reg --dataset.name otb --cs.n 20000 5000 --cs.delta 1e-5 --cs.cond_thres --train_cs > otb/snapshots/output_est_conf_set_cond_thres"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task reg --dataset.name otb --cs.n 20000 5000 --cs.delta 1e-5 --cs.no_cal --cs.cond_thres --train_cs > otb/snapshots/output_est_conf_set_no_cal_cond_thres"

