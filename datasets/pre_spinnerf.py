"""
Prepare a 'spinnerf' folder for running spinnerf
"""

import json
import os
from subprocess import check_output

from pre_nerf import parse


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path

    # Read refine params from json path
    with open(json_path, 'r') as file:
        down_factor = json.load(file)[dataset_name][scene_name]['down_factor']

    # Copy through command
    in_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name)
    out_dir = os.path.join(out_dir, f'{dataset_name}_spinnerf', scene_name)

    dirs_mapping = {
        'images': 'images',
        f'images_{down_factor}_ori': f'images_{down_factor}',
        f'images_{down_factor}/masks': f'images_{down_factor}/label',
        'sparse/0': 'sparse/0'
    }

    for in_dir_name, out_dir_name in dirs_mapping.items():
        os.makedirs(os.path.join(out_dir, out_dir_name), exist_ok=True)
        check_output('cp {}/* {}'.format(
            os.path.join(in_dir, in_dir_name), os.path.join(out_dir, out_dir_name)), shell=True)

    check_output('cp {} {}'.format(
        os.path.join(in_dir, 'poses_bounds.npy'), os.path.join(out_dir, 'poses_bounds.npy')), shell=True)


if __name__ == '__main__':
    main()
