#!/bin/bash

# GRASP_TRAIN_OBJECTS='conic_cup fountain_vase circular_table hex_deep_bowl smushed_dumbbell square_prism_bin narrow_tray colunnade_top stalagcite_chunk bongo_drum_bowl pacifier_vase beehive_funnel crooked_lid_trash_can toilet_bowl pepsi_bottle tongue_chair modern_canoe pear_ringed_vase short_handle_cup bullet_vase glass_half_gallon flat_bottom_sack_vase trapezoidal_bin vintage_canoe bathtub flowery_half_donut t_cup cookie_circular_lidless_tin box_sofa two_layered_lampshade conic_bin jar bunsen_burner long_vase ringed_cup_oversized_base aero_cylinder'
# GRASP_TEST_OBJECTS='square_rod_embellishment grill_trash_can shed sack_vase two_handled_vase thick_wood_chair curved_handle_cup baseball_cap elliptical_capsule'
GRASP_TRAIN_OBJECTS='flat_bottom_sack_vase t_cup two_layered_lampshade'
GRASP_TEST_OBJECTS='shed baseball_cap elliptical_capsule'

for obj in $GRASP_TRAIN_OBJECTS; do
    CUDA_VISIBLE_DEVICES=2 python scripts/scripted_collect_parallel.py -e Widow250GraspHard_${obj}-v0 -pl rotate_grasp -a grasp_success -n 200 -t 25 -d /media/3tb/chet/robo_exp/trajs/rotate_grasp_hard_train_successful/${obj} --target-object ${obj} -p 10
done

for obj in $GRASP_TEST_OBJECTS; do
    CUDA_VISIBLE_DEVICES=2 python scripts/scripted_collect_parallel.py -e Widow250GraspHard_${obj}-v0 -pl rotate_grasp -a grasp_success -n 200 -t 25 -d /media/3tb/chet/robo_exp/trajs/rotate_grasp_hard_test_successful/${obj} --target-object ${obj} -p 10
done