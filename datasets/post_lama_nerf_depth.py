"""
Post process after lama, including restore filenames, for nerf + depth training
"""

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pre_nerf import parse
from utils.colmap.read_write_model import read_images_binary


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    img_file_type = args.img_file_type

    # Restore filenames to match COLMAP cam params
    cam_dir = os.path.join(in_dir, f'{dataset_name}_depth', scene_name, f'sparse/0')
    images = read_images_binary(os.path.join(cam_dir, 'images.bin'))

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    imgs_dir = os.path.join(in_dir, f'{dataset_name}_depth', scene_name, f'lama_out_refine')
    img_paths = [os.path.join(imgs_dir, path) for path in os.listdir(imgs_dir) if path.endswith(img_file_type)]
    img_paths = sorted(img_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    assert len(img_paths) == len(img_names)

    out_imgs_dir = os.path.join(in_dir, f'{dataset_name}_depth', scene_name, 'depth')
    os.makedirs(out_imgs_dir, exist_ok=True)

    print(f'Restore filenames from {imgs_dir} to {out_imgs_dir}')

    i = 0
    for img_path in tqdm(img_paths):
        img_path = str(img_path)
        out_path = os.path.join(out_imgs_dir, Path(img_names[i]).name)

        # Reduce 3 channel to 1 channel
        depth = cv2.imread(img_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(out_path, depth)
        i += 1


if __name__ == '__main__':
    main()
