#!/bin/bash

ENV_TYPES='Easy Big Orient Loc Hard'

for env_type in $ENV_TYPES; do
    CUDA_VISIBLE_DEVICES=2 python scripts/scripted_collect_parallel.py -e Widow250Grasp${env_type}RandomTrain-v0 -pl grasp -a grasp_success -n 5000 -t 15 -d /media/3tb/chet/robo_exp/grasp_train_trajs/${env_type} --save-all -p 10
done