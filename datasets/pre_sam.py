"""
This file prepare a 'sparse' folder for run NeRF without delete
"""
import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from subprocess import check_output

from tqdm import tqdm

sys.path.append(os.getcwd())  # noqa
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
    for path in tqdm(os.listdir(in_dir)):
        in_path = os.path.join(in_dir, path)
        out_path = os.path.join(out_dir, path)

        shutil.copy(in_path, out_path)

    # check whether the filename in COLMAP files are consistent with filename in down-sample folder
    cameras, images, points3D = read_model(path=out_dir, ext='.bin')

    # Needs to re-write COLMAP files
    if dataset_name in filename_convert_dict and scene_name in filename_convert_dict[dataset_name]:
        images_modified = copy.deepcopy(images)
        for image_id, image in images.items():
            filestem = filename_convert_dict[dataset_name][scene_name](Path(image.name).stem)
            # noinspection PyProtectedMember
            images_modified[image_id] = \
                images_modified[image_id]._replace(name=os.path.join(Path(image.name).parent, f'{filestem}.png'))

        write_images_binary(images_modified, os.path.join(out_dir, 'images.bin'))


def handle_imgs(in_dir, out_dir, kwargs):
    # Pay attention to exif. Now we did not handle this
    # We need to check whether there is an exif transpose in the image
    # If yes, we apply the transform and re-write the image
    ori_img_dir = os.path.join(Path(in_dir).parent, 'images')

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
        img_suffix = Path(os.listdir(ori_img_dir)[0]).suffix.replace('.', '')
        down_sample_imgs(ori_img_dir, in_dir, kwargs['down_factor'], img_suffix)

    print(f'Copy img files from {in_dir} to {out_dir}')
    in_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if
                path.lower().endswith(kwargs['img_file_type'])]
    in_paths = sorted(in_paths)

    if 'num_imgs' in kwargs:
        in_paths = in_paths[kwargs['num_imgs']:]

    for in_path in tqdm(in_paths):
        out_path = os.path.join(out_dir, Path(in_path).name)
        shutil.copy(in_path, out_path)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/sam.json')

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

    # Reformat nerf_synthetic_colmap structure
    if dataset_name == 'nerf_synthetic_colmap':
        # in_img_dir = os.path.join(in_dir, dataset_name, scene_name, 'colmap_results/dense/images')
        # out_img_dir = os.path.join(in_dir, dataset_name, scene_name, f'images')
        # if not os.path.exists(out_img_dir):
        #     os.makedirs(out_img_dir)
        #     check_output('cp {}/* {}'.format(in_img_dir, out_img_dir), shell=True)

        in_cam_dir = os.path.join(in_dir, dataset_name, scene_name, 'colmap_results/dense/sparse')
        out_cam_dir = os.path.join(in_dir, dataset_name, scene_name, 'sparse/0')
        if not os.path.exists(out_cam_dir):
            os.makedirs(out_cam_dir)
            check_output('cp {}/* {}'.format(in_cam_dir, out_cam_dir), shell=True)

    # Handle images
    in_img_dir = os.path.join(in_dir, dataset_name, scene_name, f'images_{params["down_factor"]}')
    out_img_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, f'images_{params["down_factor"]}')

    os.makedirs(out_img_dir, exist_ok=True)
    handle_imgs(in_img_dir, out_img_dir, params)

    # Handle cam params
    in_cam_dir = os.path.join(in_dir, dataset_name, scene_name, 'sparse/0')
    out_cam_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, 'sparse/0')

    os.makedirs(out_cam_dir, exist_ok=True)
    handle_cams(in_cam_dir, out_cam_dir, dataset_name, scene_name)

    # Warp a video
    in_img_dir = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, f'images_{params["down_factor"]}')
    out_path = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, 'input_views.mp4')
    imgs2video(in_img_dir, out_path, params['img_file_type'], fps=10)


if __name__ == '__main__':
    main()
