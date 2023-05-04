"""
Post process after lama, including restore filenames, extract a single mask folder, warp inpainting results to video
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pre_nerf import parse
from utils.colmap.read_write_model import read_images_binary
from utils.img import imgs2video


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    img_file_type = args.img_file_type

    # Read refine params from json path
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        down_factor = json_content[dataset_name][scene_name]['down_factor']

    # Restore filenames to match COLMAP cam params
    cam_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, f'sparse/0')
    images = read_images_binary(os.path.join(cam_dir, 'images.bin'))

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    imgs_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, f'lama_out_refine')
    img_paths = [os.path.join(imgs_dir, path) for path in os.listdir(imgs_dir) if path.endswith(img_file_type)]
    img_paths = sorted(img_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    assert len(img_paths) == len(img_names)

    out_imgs_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, f'images_{down_factor}')
    os.makedirs(out_imgs_dir, exist_ok=True)

    print(f'Restore filenames from {imgs_dir} to {out_imgs_dir}')

    i = 0
    for img_path in tqdm(img_paths):
        img_path = str(img_path)
        out_path = os.path.join(out_imgs_dir, Path(img_names[i]).name)

        shutil.copy(img_path, out_path)
        i += 1

    # Extract a single mask folder from 'lama' folder
    in_mask_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, 'lama')
    in_mask_paths = [os.path.join(in_mask_dir, f) for f in os.listdir(in_mask_dir) if 'mask' in f]  # noqa
    in_mask_paths = sorted(in_mask_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    out_mask_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, f'images_{down_factor}', 'masks')
    os.makedirs(out_mask_dir, exist_ok=True)

    print(f'Extract a single mask folder from {in_mask_dir} to {out_mask_dir}')
    for in_mask_path in tqdm(in_mask_paths):
        out_mask_path = os.path.join(
            out_mask_dir, 'img{:0>3d}.png'.format(int(Path(in_mask_path).name.split('_')[0][len('image'):])))  # noqa

        mask = cv2.imread(in_mask_path)
        mask = mask[:, :, 0]

        cv2.imwrite(out_mask_path, mask)

    # Warp a video
    out_video_path = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, 'lama.mp4')
    imgs2video(imgs_dir, out_video_path, img_file_type, fps=10)


if __name__ == '__main__':
    main()
