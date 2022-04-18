#!/bin/bash

GRASP_TRAIN_OBJECTS='conic_cup fountain_vase circular_table hex_deep_bowl smushed_dumbbell square_prism_bin'

GRASP_TEST_OBJECTS='square_rod_embellishment grill_trash_can shed sack_vase'

for obj in $GRASP_TRAIN_OBJECTS; do
    python scripts/scripted_collect_parallel.py -e Widow250GraspEasyTrain-${obj}-v0 -pl grasp -a grasp_success -n 5000 -t 15 -d /media/3tb/chet/robo_exp/easy_train/${obj} --target-object ${obj} -p 8
done

for obj in $GRASP_TEST_OBJECTS; do
    GPU3 python scripts/scripted_collect_parallel.py -e Widow250GraspEasyTest-${obj}-v0 -pl grasp -a grasp_success -n 5000 -t 15 -d /media/3tb/chet/robo_exp/easy_test/${obj} --target-object ${obj} -p 8
done