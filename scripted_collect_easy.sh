#!/bin/bash

GRASP_TRAIN_OBJECTS='conic_cup fountain_vase circular_table hex_deep_bowl smushed_dumbbell square_prism_bin narrow_tray colunnade_top'

for obj in $GRASP_TRAIN_OBJECTS; do
    CUDA_VISIBLE_DEVICES=3 python scripts/scripted_collect_parallel.py -e Widow250GraspEasy_${obj}-v0 -pl grasp -a grasp_success -n 5000 -t 15 -d /media/3tb/chet/robo_exp/grasp_easy_trajs/${obj} --target-object ${obj} -p 10
    rm -R -- /media/3tb/chet/robo_exp/grasp_easy_trajs/*/
done