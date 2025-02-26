import argparse
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-l', '--lines', type=int, default=-1)
    args = parser.parse_args()

    input_files = args.input

    all_files = []
    print('loading..')
    for f in tqdm(input_files):
        print(f)
        data = np.load(f, allow_pickle=True)
        if args.lines >= 0:
            data = data[:args.lines]
        all_files.append(data)
    print('concatenating..')
    all_data = np.concatenate(all_files, axis=0)
    print('saving..')
    np.savez_compressed(args.output, all_data)
