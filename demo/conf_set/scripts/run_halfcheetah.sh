#!/bin/sh
SNAPSHOT_ROOT=halfcheetah/snapshots
mkdir -p $SNAPSHOT_ROOT
# run a full model first to share the calibrated results
CUDA_VISIBLE_DEVICES=1 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --train_cs > $SNAPSHOT_ROOT/output_est_conf_set_ours
screen -dm bash -c "CUDA_VISIBLE_DEVICES=2 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.no_cal --train_cs > $SNAPSHOT_ROOT/output_est_conf_set_no_cal"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=0 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.no_db --train_cs > $SNAPSHOT_ROOT/output_est_conf_set_no_db"
screen -dm bash -c "CUDA_VISIBLE_DEVICES=3 python3 pac_conf_set.py --task rl --dataset.name halfcheetah --cs.n 5000 --cs.no_acc --cal.n_epochs 2000 --train_cs > $SNAPSHOT_ROOT/output_est_conf_set_no_acc"

