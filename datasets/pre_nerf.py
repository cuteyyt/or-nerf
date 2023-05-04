"""
Prepare a 'sparse' folder for running scenes without delete and processing sam
"""
import argparse
import copy
import json
import os
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
from tqdm import tqdm

from utils.colmap.read_write_model import read_model, write_images_binary
from utils.img import imgs2video, down_sample_imgs

filename_convert_dict = {
    'nerf_llff_data': {
        'fortress': lambda x: 'image' + '{:0>3d}'.format(int(x.split('_')[1]) - 1800),
    },
    'llff_real_iconic': {
        'data5_piano': lambda x: 'image' + '{:0>3d}'.format(int(x.split('_')[1]) - 2096),
    }
}


# noinspection PyPep8Naming
def handle_cams(in_dir, out_dir, dataset_name, scene_name):
    print(f'Copy cam params files from {in_dir} to {out_dir}')

    # Copy first
    check_output(f'cp {os.path.join(in_dir, "cameras.bin")} {os.path.join(out_dir, "cameras.bin")}', shell=True)
    check_output(f'cp {os.path.join(in_dir, "images.bin")} {os.path.join(out_dir, "images.bin")}', shell=True)
    check_output(f'cp {os.path.join(in_dir, "points3D.bin")} {os.path.join(out_dir, "points3D.bin")}', shell=True)

    # check whether the filename in COLMAP files are consistent with filename in down-sample folder
    cameras, images, points3D = read_model(path=out_dir, ext='.bin')

    # Needs to re-write COLMAP files
    print(f'Modify images.bin to match image name and its cam params')
    images_modified = copy.deepcopy(images)
    for image_id, image in tqdm(images.items()):
        if dataset_name in filename_convert_dict and scene_name in filename_convert_dict[dataset_name]:
            filestem = filename_convert_dict[dataset_name][scene_name](Path(image.name).stem)
        else:
            filestem = Path(image.name).stem
        # noinspection PyProtectedMember
        images_modified[image_id] = \
            images_modified[image_id]._replace(name=os.path.join(Path(image.name).parent, f'{filestem}.png'))

        write_images_binary(images_modified, os.path.join(out_dir, 'images.bin'))

    names = [images_modified[k].name for k in images_modified]
    names = np.sort(names)
    return names


def handle_imgs(in_ori_dir, out_ori_dir, kwargs):
    # Pay attention to exif info. Now we did not handle this
    # We need to check whether there is an exif transpose in the image
    # If yes, we apply the transform and re-write the image

    print(f'Copy img files from {Path(in_ori_dir).parent} to {Path(out_ori_dir).parent}')

    in_ori_paths = [os.path.join(in_ori_dir, path) for path in os.listdir(in_ori_dir) if
                    path.lower().endswith(kwargs['img_file_type'])]
    in_ori_paths = sorted(in_ori_paths)

    if 'img_indices' in kwargs:
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


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/sam_points.json')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


# noinspection SpellCheckingInspection
def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    params = {'img_file_type': args.img_file_type}

    # Read config json
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        params_json = json_content[dataset_name][scene_name]
        params = dict(params, **params_json)

    if dataset_name == 'spinnerf_dataset':
        params['img_indices'] = 40

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
