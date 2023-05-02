import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mask_refine import mask_refine
from pre_sam import parse
from utils.colmap.read_write_model import read_images_binary


def pre_lama(args):
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    # img_file_type = args.img_file_type

    in_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name)
    out_dir = os.path.join(out_dir, f'{dataset_name}_sam', scene_name)
    print(f'Prepare lama dataset from {in_dir} to {out_dir}')

    # Read refine params from json path
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        refine_params = json_content[dataset_name][scene_name]['refine_params']
        down_factor = json_content[dataset_name][scene_name]['down_factor']

    imgs_dir = os.path.join(in_dir, f'images_{down_factor}_ori')
    masks_dir = os.path.join(in_dir, 'masks')

    cam_dir = os.path.join(in_dir, 'sparse/0')
    images = read_images_binary(os.path.join(cam_dir, 'images.bin'))

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    # img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if path.endswith(img_file_type)]
    rgb_paths = [os.path.join(imgs_dir, f) for f in img_names]

    out_lama_dir = os.path.join(out_dir, 'lama')
    out_rgb_masked_dir = os.path.join(out_dir, 'lama_masked')

    os.makedirs(out_lama_dir, exist_ok=True)
    os.makedirs(out_rgb_masked_dir, exist_ok=True)

    with tqdm(total=len(rgb_paths)) as t_bar:
        for i, rgb_path in enumerate(rgb_paths):
            filename = Path(rgb_path).name
            file_id = i

            mask_path = os.path.join(masks_dir, filename)

            out_img_path = os.path.join(out_lama_dir, 'image{:0>3d}.png'.format(file_id))
            out_mask_path = os.path.join(out_lama_dir, 'image{:0>3d}_mask{:0>3d}.png'.format(file_id, file_id))
            out_img_masked_path = os.path.join(out_rgb_masked_dir, filename)

            shutil.copy(rgb_path, out_img_path)

            # Refine mask files
            mask = cv2.imread(mask_path)
            mask_refined = mask_refine(mask.copy(), refine_params)

            cv2.imwrite(out_mask_path, mask_refined)

            # Write img with masks
            img = cv2.imread(rgb_path)
            rgb_masked = np.copy(img)

            rgb_masked[mask_refined == 255] = [255, 255, 255]
            cv2.imwrite(out_img_masked_path, rgb_masked)

            t_bar.update(1)


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
