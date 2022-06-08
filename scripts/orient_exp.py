import subprocess
from roboverse.assets.shapenet_object_lists import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    args = parser.parse_args()

    for obj_name in GRASP_TRAIN_OBJECTS + GRASP_TEST_OBJECTS:
        for orient_num in range(6):
            command = [
                'python',
                'scripts/scripted_collect_parallel.py',
                '-pl',args.policy_name,
                '-a','grasp_success',
                '-e',f'Widow250GraspBig_{obj_name}_orient_{orient_num}-v0',
                '-n','50',
                '-t','50',
                '--target-object',obj_name,
                '-d',f'/nfs/kun2/users/chet/robo_exp/orient_trajs/{obj_name}_orient_{orient_num}_{args.policy_name}/',
                '--save-all',
            ]
            subproc = subprocess.Popen(command)
            time.sleep(1)
            subproc.wait()
