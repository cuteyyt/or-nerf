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

    try:
        scaled_poly = np.asarray(scaled_poly)
    except Exception as e:
        scaled_poly = [np.array(item) for item in scaled_poly]
    return scaled_poly


def mask_refine(mask, **kwargs):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Step 0. Filter irregular masks
    mask_filtered = np.zeros_like(mask_gray)
    if 'min_contour_size' in kwargs:
        contours, hierarchies = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Only keep the contour with the mask area
        if len(contours) > 1:
            contours_refined = list()
            for i, contour in enumerate(contours):
                contour_area = cv2.contourArea(contour)
                print(contour_area)
                if contour_area > kwargs['min_contour_size']:
                    contours_refined.append(contour)
        else:
            contours_refined = contours

        cv2.drawContours(mask_filtered, contours_refined, -1, (255, 255, 255), cv2.FILLED)

    # Step 1. Dilate to smooth the borders, fill-in holes
    dilate_kernel_size = kwargs['dilate_kernel_size']
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))

    mask_dilated = np.copy(mask_filtered)
    contours, hierarchies = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if 'dilate_iters' in kwargs:
        num_contours = len(contours)
        while num_contours > 1:
            #print(num_contours)
            mask_dilated = cv2.dilate(mask_dilated, dilate_kernel, iterations=kwargs['dilate_iters'])
            contours, hierarchies = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)

    # Step 2. Scale the contour
    mask_scaled = np.zeros_like(mask_dilated)
    if 'contour_scale_size' in kwargs:
        contour_scaled = scale_by_pyclipper(contours[0].squeeze(1), scale_size=kwargs['contour_scale_size'])
        cv2.drawContours(mask_scaled, contour_scaled, -1, (255, 255, 255), -1)

    # mask_refined = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)
    mask_refined = mask_scaled.copy()

    return mask_refined


def pre_lama(args):
    in_dir = os.path.join(args.in_dir, f'{args.dataset_name}_sam', args.scene_name)
    out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_sam', args.scene_name)
    print(f'Prepare lama dataset from {in_dir} to {out_dir}')

    # Read refine params from json path
    with open(args.mask_refine_json, 'r') as file:
        lama_params = json.load(file)
        refine_params = lama_params[args.dataset_name][args.scene_name]['refine_params']
        args.down_factor = lama_params[args.dataset_name][args.scene_name]['down_factor']

    imgs_dir = os.path.join(in_dir, f'images_{args.down_factor}_ori')
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
            #print(mask_path)

            out_img_path = os.path.join(out_lama_dir, 'image{:0>3d}.png'.format(file_id))
            out_mask_path = os.path.join(out_lama_dir, 'image{:0>3d}_mask{:0>3d}.png'.format(file_id, file_id))
            out_img_masked_path = os.path.join(out_rgb_masked_dir, filename)

            # Copy mask files directly
            shutil.copy(rgb_path, out_img_path)

            # Refine mask files

            mask = cv2.imread(mask_path)
            if args.mask_refine:
                mask_refined = mask_refine(mask.copy(), **refine_params)
            else:
                mask_refined = mask.copy()[:,:,2]
            cv2.imwrite(out_mask_path, mask_refined)

            # Write img with masks
            img = cv2.imread(rgb_path)
            rgb_masked = np.copy(img)

            rgb_masked[mask_refined == 255] = [255, 255, 255]


            cv2.imwrite(out_img_masked_path, rgb_masked)

            t_bar.update(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='', type=str, help='in data dir, containing images and masks')
    parser.add_argument('--out_dir', default='', type=str, help='lama input dir')
    parser.add_argument('--dataset_name', type=str, default='', help='')
    parser.add_argument('--scene_name', type=str, default='', help='')

    parser.add_argument('--mask_refine_json', default='', type=str, help='mask refine params json path')
    parser.add_argument('--down_factor', type=float, default=4, help='img resolution down scale factor')
    parser.add_argument('--mask_refine', '-R',default=True, action='store_false')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
