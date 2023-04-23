"""
This file prepare a 'sparse' folder for run NeRF without delete
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from subprocess import check_output

from tqdm import tqdm


def handle_cams(in_dir, out_dir):
    print(f'Copy cam params files from {in_dir} to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    for path in tqdm(os.listdir(in_dir)):
        in_path = os.path.join(in_dir, path)
        out_path = os.path.join(out_dir, path)

        shutil.copy(in_path, out_path)


def down_sample_imgs(src_dir, target_dir, down_factor, img_suffix):
    print('Minifying', down_factor, (Path(src_dir)).parent.name)

    cwd = os.getcwd()
    resize_arg = '{}%'.format(100. / down_factor)

    os.makedirs(target_dir)
    check_output('cp {}/* {}'.format(src_dir, target_dir), shell=True)

    args = ' '.join(['mogrify', '-resize', resize_arg, '-format', 'png', '*.{}'.format(img_suffix)])
    print(args)
    os.chdir(target_dir)
    check_output(args, shell=True)
    os.chdir(cwd)

    if img_suffix != 'png':
        check_output('rm {}/*.{}'.format(target_dir, img_suffix), shell=True)
        print('Removed duplicates')
    print('Done')


def handle_imgs(in_dir, out_dir, **kwargs):
    if not os.path.exists(in_dir):
        img_suffix = Path(os.path.join(in_dir, os.listdir(in_dir)[0])).suffix
        src_dir = f'{Path(in_dir).parent}/{Path(in_dir).name.split("_")[0]}'
        down_sample_imgs(src_dir, in_dir, kwargs['down_factor'], img_suffix)

    print(f'Copy img files from {in_dir} to {out_dir}')
    in_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir)
                if path.lower().endswith(kwargs['img_file_type'])]
    in_paths = sorted(in_paths)

    if 'num_imgs' in kwargs:
        in_paths = in_paths[:kwargs['num_imgs']]

    os.makedirs(out_dir, exist_ok=True)
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

    if dataset_name == 'nerf_llff_data':
        # Down sample images to desired resolution
        in_img_dir = os.path.join(in_dir, dataset_name, scene_name, f'images_{params["down_factor"]}')
        out_img_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, f'images_{params["down_factor"]}')
        handle_imgs(in_img_dir, out_img_dir, **params)

        # Handle cam params
        in_cam_dir = os.path.join(in_dir, dataset_name, scene_name, 'sparse/0')
        out_cam_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, 'sparse/0')
        handle_cams(in_cam_dir, out_cam_dir)

    elif dataset_name == 'spinnerf_dataset':
        in_img_dir = os.path.join(in_dir, dataset_name, scene_name, f'images_{params["down_factor"]}')
        out_img_dir = os.path.join(out_dir, f'{dataset_name}_sparse', scene_name, f'images_{params["down_factor"]}')
        params['num_imgs'] = 60
        handle_imgs(in_img_dir, out_img_dir, **params)

    elif dataset_name == 'ibrnet_dataset':
        pass
    elif dataset_name == 'dmsr':
        pass
    elif dataset_name == 'nerf_synthetic':
        pass
    elif dataset_name == 'nerf_real_360':
        pass
    elif dataset_name == 'mip_360_dataset':
        pass


if __name__ == '__main__':
    main()
