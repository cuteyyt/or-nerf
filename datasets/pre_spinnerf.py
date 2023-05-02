"""
Copy files from *_sam folder to run spinnerf
"""
import json
import os
from subprocess import check_output

import cv2
from tqdm import tqdm

from pre_sam import parse


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path

    # Read refine params from json path
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        down_factor = json_content[dataset_name][scene_name]['down_factor']

    # Copy through command
    in_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name)
    out_dir = os.path.join(out_dir, f'{dataset_name}_spinnerf', scene_name)

    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, f'images_{down_factor}'), exist_ok=True)

    check_output('cp {}/* {}'.format(os.path.join(in_dir, 'images'), os.path.join(out_dir, 'images')), shell=True)
    check_output('cp {}/* {}'.format(
        os.path.join(in_dir, f'images_{down_factor}'), os.path.join(out_dir, f'images_{down_factor}')), shell=True)
    check_output('cp {} {}'.format(
        os.path.join(in_dir, 'poses_bounds.npy'), os.path.join(out_dir, 'poses_bounds.npy')), shell=True)

    # Gen spinnerf format masks
    in_mask_dir = os.path.join(in_dir, 'lama')
    in_mask_paths = [os.path.join(in_mask_dir, f) for f in os.listdir(in_mask_dir) if 'mask' in f]

    out_mask_dir = os.path.join(out_dir, f'images_{down_factor}', 'label')
    os.makedirs(out_mask_dir, exist_ok=True)

    i = 0
    for in_mask_path in tqdm(in_mask_paths):
        out_mask_path = os.path.join(out_mask_dir, 'img{:0>3d}.png'.format(i))

        mask = cv2.imread(in_mask_path)
        mask = mask[:, :, 0]
        mask[mask == 255] = 1

        cv2.imwrite(out_mask_path, mask)
        i += 1


if __name__ == '__main__':
    main()
