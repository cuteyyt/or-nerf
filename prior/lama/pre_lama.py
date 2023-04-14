import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyclipper
from tqdm import tqdm


def scale_by_pyclipper(points, scale_size=1.):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # JT_ROUND
    scaled_poly = pco.Execute(int(scale_size))
    pco.Clear()

    return np.asarray(scaled_poly)


def mask_refine(mask, **kwargs):
    # Step 1. Dilate to smooth the borders, fill-in holes
    dilate_kernel_size = kwargs['dilate_kernel_size']
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    mask_dilated = cv2.dilate(mask, dilate_kernel, iterations=kwargs['dilate_iters'])

    # Step 2. Scale the contour
    # Find contours first
    mask_gray = cv2.cvtColor(mask_dilated, cv2.COLOR_BGR2GRAY)

    contours, hierarchies = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Scale the contour polygon
    mask_scaled = np.zeros_like(mask_gray)
    for contour in contours:
        if len(contour.squeeze(1)) > 2:
            contour_scaled = scale_by_pyclipper(contour.squeeze(1), scale_size=kwargs['contour_scale_size'])
            cv2.fillPoly(mask_scaled, contour_scaled, (255, 255, 255))

    mask_refined = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)

    return mask_refined


def pre_lama(args):
    in_dir = os.path.join(args.in_dir, f'{args.dataset_name}_sam', args.scene_name)
    out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_sam', args.scene_name)
    print(f'Prepare lama dataset from {in_dir} to {out_dir}')

    imgs_dir = os.path.join(in_dir, 'imgs_ori')
    masks_dir = os.path.join(in_dir, 'masks')

    rgb_paths = [os.path.join(imgs_dir, path) for path in os.listdir(imgs_dir) if
                 path.lower().endswith(args.img_file_type)]
    rgb_paths = sorted(rgb_paths)

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

            # Copy mask files directly
            shutil.copy(rgb_path, out_img_path)

            # Refine mask files
            # Read refine params from json path
            with open(args.mask_refine_json, 'r') as file:
                refine_params = json.load(file)
                refine_params = refine_params[args.dataset_name][args.scene_name]['refine_params']

            mask = cv2.imread(mask_path)
            mask_refined = mask_refine(mask.copy(), **refine_params)
            cv2.imwrite(out_mask_path, mask_refined)

            # Write img with masks
            img = cv2.imread(rgb_path)
            rgb_masked = np.copy(img)

            mask_refined_gray = cv2.cvtColor(mask_refined, cv2.COLOR_BGR2GRAY)
            rgb_masked[mask_refined_gray == 255] = [255, 255, 255]

            cv2.imwrite(out_img_masked_path, rgb_masked)

            t_bar.update(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='', type=str, help='in data dir, containing images and masks')
    parser.add_argument('--out_dir', default='', type=str, help='lama input dir')
    parser.add_argument('--dataset_name', type=str, default='', help='')
    parser.add_argument('--scene_name', type=str, default='', help='')

    parser.add_argument('--mask_refine_json', default='', type=str, help='mask refine params json path')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
