import argparse
import numpy as np
from collections import defaultdict
import glob
from pathlib import Path

def convert_dataset_d4rl(src_path, dest_path):
    trajs = np.load(src_path, allow_pickle=True)
    traj_keys = trajs[0].keys()
    dataset_dict = defaultdict(list)
    for k in traj_keys:
        val_is_dict = isinstance(trajs[0][k][0], dict)
        if val_is_dict:
            dataset_dict[k] = defaultdict(list)
        for traj in trajs:
            for val in traj[k]:
                if val_is_dict:
                    for sub_key, sub_val in val.items():
                        dataset_dict[k][sub_key].append(sub_val)
                else:
                    dataset_dict[k].append(val)
        if val_is_dict:
            for sub_key, val_list in dataset_dict[k].items():
                dataset_dict[k][sub_key] = np.array(val_list)
        else:
            dataset_dict[k] = np.array(dataset_dict[k])
    np.save(dest_path, dataset_dict)

def convert_d4rl(file_path, dest_dir_path):
    dest_dir = Path(dest_dir_path)
    if not dest_dir.exists():
        dest_dir.mkdir()

    for input_file in input_files:
        if input_file.endswith('npy'):
            src_path = Path(input_file)
            dest_path = Path(f'{dest_dir_path}/{src_path.stem}_d4rl{src_path.suffix}')
            convert_dataset_d4rl(src_path, dest_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    input_files = args.input
    dest_dir = args.output
    convert_d4rl(input_files, dest_dir)


