#!/bin/bash

while getopts e:n:g:d: flag
do
    case "${flag}" in
        e) env=${OPTARG};;
        n) num_trajs=${OPTARG};;
        g) gpu=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

GRASP_TRAIN_OBJECTS='conic_cup fountain_vase circular_table hex_deep_bowl smushed_dumbbell square_prism_bin narrow_tray colunnade_top stalagcite_chunk bongo_drum_bowl pacifier_vase beehive_funnel crooked_lid_trash_can toilet_bowl pepsi_bottle tongue_chair modern_canoe pear_ringed_vase short_handle_cup bullet_vase glass_half_gallon flat_bottom_sack_vase trapezoidal_bin vintage_canoe bathtub flowery_half_donut t_cup cookie_circular_lidless_tin box_sofa two_layered_lampshade conic_bin jar bunsen_burner long_vase ringed_cup_oversized_base aero_cylinder'
GRASP_TEST_OBJECTS='square_rod_embellishment grill_trash_can shed sack_vase two_handled_vase thick_wood_chair curved_handle_cup baseball_cap elliptical_capsule'

for obj in $GRASP_TEST_OBJECTS; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/scripted_collect_parallel.py -e Widow250Grasp${env}_${obj}-v0 -pl rotate_grasp -a grasp_success -n $num_trajs -t 50 -d ${dir}/grasp_${env,,}_test_successful/${obj} --target-object ${obj} -p 10
done

for obj in $GRASP_TRAIN_OBJECTS; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/scripted_collect_parallel.py -e Widow250Grasp${env}_${obj}-v0 -pl rotate_grasp -a grasp_success -n $num_trajs -t 50 -d ${dir}/grasp_${env,,}_train_successful/${obj} --target-object ${obj} -p 10
done