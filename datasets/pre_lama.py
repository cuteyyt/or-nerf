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
    pco.AddPath(points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)  # JT_ROUND
    scaled_poly = pco.Execute(int(scale_size))
    pco.Clear()

    scaled_poly = np.asarray(scaled_poly)
    return scaled_poly


def mask_refine(mask, **kwargs):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Step 1. Filter irregular masks

    if 'num_masks' in kwargs:
        mask_filtered = np.zeros_like(mask_gray)
        contours, hierarchies = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = list()
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            contour_areas.append(contour_area)
        contour_areas = np.asarray(contour_areas)

        contours_refined = list()
        contours_indices = np.argsort(contour_areas)[-1::-1][:kwargs['num_masks']]
        for idx in contours_indices:
            contours_refined.append(contours[idx])

        cv2.drawContours(mask_filtered, contours_refined, -1, (255, 255, 255), cv2.FILLED)
    else:
        mask_filtered = np.copy(mask_gray)

    # Step 2. Dilate to smooth the borders, fill-in holes
    mask_dilated = np.copy(mask_filtered)
    if 'dilate_kernel_size' and 'dilate_iters' in kwargs:
        dilate_kernel_size = kwargs['dilate_kernel_size']
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        mask_dilated = cv2.dilate(mask_dilated, dilate_kernel, iterations=kwargs['dilate_iters'])

    # Step 3. Scale the contour
    if 'contour_scale_size' in kwargs:
        mask_scaled = np.zeros_like(mask_dilated)
        contours, hierarchies = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) == kwargs['num_masks']

        for contour in contours:
            contour_scaled = scale_by_pyclipper(contour.squeeze(1), scale_size=kwargs['contour_scale_size'])
            cv2.drawContours(mask_scaled, contour_scaled, -1, (255, 255, 255), -1)
    else:
        mask_scaled = np.copy(mask_dilated)

    mask_refined = mask_scaled.copy()
    return mask_refined


def pre_lama(args):
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    img_file_type = args.img_file_type

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

    rgb_paths = [os.path.join(imgs_dir, path) for path in os.listdir(imgs_dir) if path.endswith(img_file_type)]
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

            shutil.copy(rgb_path, out_img_path)

            # Refine mask files
            mask = cv2.imread(mask_path)
            mask_refined = mask_refine(mask.copy(), **refine_params)

            cv2.imwrite(out_mask_path, mask_refined)

            # Write img with masks
            img = cv2.imread(rgb_path)
            rgb_masked = np.copy(img)

            rgb_masked[mask_refined == 255] = [255, 255, 255]
            cv2.imwrite(out_img_masked_path, rgb_masked)

            t_bar.update(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--json_path', type=str)

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
