"""
Prepare a 'sparse' folder for running scenes without delete and processing sam
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.img import imgs2video, down_sample_imgs


def handle_imgs(in_ori_dir, out_ori_dir, kwargs):
    # Pay attention to exif info. Now we did not handle this
    # We need to check whether there is an exif transpose in the image
    # If yes, we apply the transform and re-write the image

    print(f'Copy img files from {Path(in_ori_dir).parent} to {Path(out_ori_dir).parent}')

    in_ori_paths = [os.path.join(in_ori_dir, path) for path in os.listdir(in_ori_dir) if
                    path.lower().endswith(kwargs['img_file_type'])]
    in_ori_paths = sorted(in_ori_paths)

    if 'img_indices' in kwargs and not kwargs['is_test']:
        in_ori_paths = in_ori_paths[kwargs['img_indices']:]

    i = 0
    for in_ori_path in tqdm(in_ori_paths):
        out_path = os.path.join(out_ori_dir, 'img{:0>3d}.png'.format(i))
        img = cv2.imread(in_ori_path)
        cv2.imwrite(out_path, img)

        i += 1

    out_dir = os.path.join(Path(out_ori_dir).parent, f'images_{kwargs["down_factor"]}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        down_sample_imgs(out_ori_dir, out_dir, kwargs['down_factor'], img_suffix='png')

        if kwargs['is_test']:
            paths = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if
                     f.endswith(kwargs['img_file_type'])]
            gt_paths = paths[:kwargs['img_indices']]

            tgt_dir = os.path.join(Path(out_ori_dir).parent, f'images_gt')
            os.makedirs(tgt_dir, exist_ok=True)

            for gt_path in gt_paths:
                shutil.move(gt_path, os.path.join(tgt_dir, Path(gt_path).name))


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/sam_points.json')
    parser.add_argument('--is_test', action='store_true')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


# noinspection SpellCheckingInspection
def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    is_test = args.is_test
    params = {'img_file_type': args.img_file_type, 'is_test': args.is_test}

    if is_test or dataset_name == 'spinnerf_dataset':
        params['img_indices'] = 40

    # Read config json
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        params_json = json_content[dataset_name][scene_name]
        params = dict(params, **params_json)

    in_ori_img_dir = os.path.join(in_dir, dataset_name, scene_name, 'images')
    out_ori_img_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, 'images')

    os.makedirs(out_ori_img_dir, exist_ok=True)
    handle_imgs(in_ori_img_dir, out_ori_img_dir, params)

    # Warp a video
    in_img_dir = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, f'images_{params["down_factor"]}')
    out_path = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, 'input_views.mp4')
    imgs2video(in_img_dir, out_path, params['img_file_type'], fps=10)


if __name__ == '__main__':
    main()
