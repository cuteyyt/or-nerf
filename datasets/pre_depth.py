"""
Prepare a 'depth' folder for running nerf with lama priors
"""

import argparse
import json
import os
import shutil
from subprocess import check_output

from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scene', type=str)

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/lama.json')
    parser.add_argument('--is_test', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse()
    log_dir, in_dir, out_dir = args.log_dir, args.in_dir, args.out_dir
    dataset, scene = args.dataset, args.scene
    json_path = args.json_path
    is_test = args.is_test

    # Read resolution down-factor
    with open(json_path, 'r') as f:
        json_content = json.load(f)
        down_factor = json_content[dataset][scene]['down_factor']

    # Register related paths and copy necessary files
    # log_dir = os.path.join(log_dir, scene)
    depth_dir = os.path.join(out_dir, f'{dataset}_depth', scene)
    data_dir = os.path.join(in_dir, f'{dataset}_sam', scene)

    dirs_mapping = {
        'images': 'images',
        f'images_{down_factor}': f'images_{down_factor}',
        'sparse/0': 'sparse/0'
    }

    for in_dir_name, out_dir_name in dirs_mapping.items():
        os.makedirs(os.path.join(depth_dir, out_dir_name), exist_ok=True)
        check_output('cp -r {}/* {}'.format(
            os.path.join(data_dir, in_dir_name), os.path.join(depth_dir, out_dir_name)), shell=True)

    check_output('rm -rf {}'.format(os.path.join(depth_dir, 'label')), shell=True)
    os.makedirs(os.path.join(depth_dir, 'label'))
    check_output('mv {}/* {}'.format(
        os.path.join(depth_dir, f'images_{down_factor}/masks'), os.path.join(depth_dir, 'label')), shell=True)
    check_output('rm -rf {}'.format(os.path.join(depth_dir, f'images_{down_factor}/masks')), shell=True)

    check_output('cp {} {}'.format(
        os.path.join(data_dir, 'poses_bounds.npy'), os.path.join(depth_dir, 'poses_bounds.npy')), shell=True)

    # in_depth_dir = os.path.join(log_dir, 'render_all_200000_depth')
    in_depth_dir = log_dir
    in_depth_paths = [os.path.join(in_depth_dir, f) for f in sorted(os.listdir(in_depth_dir))]

    out_depth_dir = os.path.join(depth_dir, f'depth_ori')
    os.makedirs(out_depth_dir, exist_ok=True)

    print(f'Copy depth files from {in_depth_dir} for {out_depth_dir}')
    i = 0

    if is_test:
        i = 40

    for in_depth_path in tqdm(in_depth_paths):
        out_depth_path = os.path.join(out_depth_dir, 'img{:0>3d}.png'.format(i))

        shutil.copy(in_depth_path, out_depth_path)
        i += 1


if __name__ == '__main__':
    main()
