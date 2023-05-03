import argparse
import json
import os
import shutil
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scene', type=str)

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/lama.json')

    args = parser.parse_args()
    return args


def main():
    args = parse()
    log_dir, data_dir = args.log_dir, args.data_dir
    dataset, scene = args.dataset, args.scene
    json_path = args.json_path

    # Read resolution down-factor
    with open(json_path, 'r') as f:
        json_content = json.load(f)
        down_factor = json_content[dataset][scene]['down_factor']

    # Register related paths and copy necessary files
    log_dir = os.path.join(log_dir, scene)
    data_dir = os.path.join(data_dir, f'{dataset}_sam', scene)
    depth_dir = os.path.join(data_dir, f'{dataset}_depth', scene)

    depth_sub_dirs = {
        'images': os.path.join(depth_dir, 'images'),
        f'images_{down_factor}': os.path.join(depth_dir, f'images_{down_factor}'),
        'sparse/0': os.path.join(depth_dir, 'sparse/0')
    }

    for dir_name, depth_sub_dir in depth_sub_dirs.items():
        os.makedirs(depth_sub_dir, exist_ok=True)
        check_output(
            'cp {}/* {}'.format(os.path.join(data_dir, dir_name), depth_sub_dir), shell=True)

    check_output('cp {} {}'.format(
        os.path.join(data_dir, 'poses_bounds.npy'), os.path.join(depth_dir, 'poses_bounds.npy')), shell=True)

    # Gen masks
    in_mask_dir = os.path.join(data_dir, 'lama')
    out_mask_dir = os.path.join(depth_dir, 'masks')

    in_mask_paths = [os.path.join(in_mask_dir, f) for f in os.listdir(in_mask_dir) if 'mask' in f]
    in_mask_paths = sorted(in_mask_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    os.makedirs(out_mask_dir, exist_ok=True)
    i = 0

    for in_mask_path in tqdm(in_mask_paths):
        out_mask_path = os.path.join(out_mask_dir, 'img{:0>3d}'.format(i))
        shutil.copy(in_mask_path, out_mask_path)

        i += 1

    # Gen depths for inpainting
    in_depth_dir = os.path.join(log_dir, 'render_all_path_200000_depth')
    out_depth_ori_dir = os.path.join(data_dir, 'depth_ori')

    in_depth_paths = [os.path.join(in_depth_dir, f) for f in os.listdir(in_depth_dir)]
    in_depth_paths = sorted(in_depth_paths)

    os.makedirs(out_depth_ori_dir, exist_ok=True)
    for in_depth_path in tqdm(in_depth_paths):
        out_depth_ori_path = os.path.join(out_depth_ori_dir, Path(in_depth_path).name)

        depth = np.load(in_depth_path)
        depth_map = depth / np.max(depth)
        depth_img = (depth_map * 255).astype(np.uint8)

        cv2.imwrite(out_depth_ori_path, depth_img)


if __name__ == '__main__':
    main()
