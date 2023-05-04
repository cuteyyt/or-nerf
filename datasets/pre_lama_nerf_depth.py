"""
Prepare a lama format input folder for nerf + depth training
"""

import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pre_nerf import parse
from utils.colmap.read_write_model import read_images_binary


def pre_lama(args):
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name

    in_dir = os.path.join(in_dir, f'{dataset_name}_depth', scene_name)
    out_dir = os.path.join(out_dir, f'{dataset_name}_depth', scene_name)
    print(f'Prepare lama dataset from {in_dir} to {out_dir}')

    imgs_dir = os.path.join(in_dir, f'depth_ori')
    masks_dir = os.path.join(in_dir, 'label')

    cam_dir = os.path.join(in_dir, 'sparse/0')
    images = read_images_binary(os.path.join(cam_dir, 'images.bin'))

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    rgb_paths = [os.path.join(imgs_dir, f) for f in img_names]

    out_lama_dir = os.path.join(out_dir, 'lama')
    os.makedirs(out_lama_dir, exist_ok=True)

    with tqdm(total=len(rgb_paths)) as t_bar:
        for i, rgb_path in enumerate(rgb_paths):
            filename = Path(rgb_path).name
            file_id = i

            mask_path = os.path.join(masks_dir, filename)

            out_img_path = os.path.join(out_lama_dir, 'image{:0>3d}.png'.format(file_id))
            out_mask_path = os.path.join(out_lama_dir, 'image{:0>3d}_mask{:0>3d}.png'.format(file_id, file_id))

            shutil.copy(rgb_path, out_img_path)
            shutil.copy(mask_path, out_mask_path)

            t_bar.update(1)


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
