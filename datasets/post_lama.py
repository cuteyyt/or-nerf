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

from pre_lama import parse
from utils.colmap.read_write_model import read_images_binary
from utils.img import imgs2video


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    img_file_type, sfx = args.img_file_type, args.sfx
    is_test, is_depth = args.is_test, args.is_depth

    # Read refine params from json path
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        down_factor = json_content[dataset_name][scene_name]['down_factor']

    # Restore filenames to match COLMAP cam params
    cam_dir = os.path.join(in_dir, f'{dataset_name}_{sfx}', scene_name, f'sparse/0')
    images = read_images_binary(os.path.join(cam_dir, 'images.bin'))

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    if is_test:
        img_names = img_names[40:]

    imgs_dir = os.path.join(in_dir, f'{dataset_name}_{sfx}', scene_name, f'lama_out_refine')
    img_paths = [os.path.join(imgs_dir, path) for path in os.listdir(imgs_dir) if path.endswith(img_file_type)]
    img_paths = sorted(img_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    assert len(img_paths) == len(img_names)

    if is_depth:
        out_imgs_dir = os.path.join(in_dir, f'{dataset_name}_{sfx}', scene_name, 'depth')
    else:
        out_imgs_dir = os.path.join(in_dir, f'{dataset_name}_{sfx}', scene_name, f'images_{down_factor}')
    os.makedirs(out_imgs_dir, exist_ok=True)

    print(f'Restore filenames from {imgs_dir} to {out_imgs_dir}')

    i = 0
    for img_path in tqdm(img_paths):
        img_path = str(img_path)
        out_path = os.path.join(out_imgs_dir, Path(img_names[i]).name)

        if is_depth:
            depth = cv2.imread(img_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_path, depth)
        else:
            shutil.copy(img_path, out_path)

        i += 1

    # Warp a video
    out_video_path = os.path.join(in_dir, f'{dataset_name}_{sfx}', scene_name, 'lama.mp4')
    imgs2video(imgs_dir, out_video_path, img_file_type, fps=10)


if __name__ == '__main__':
    main()
