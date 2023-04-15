import argparse
import json
import os
import shutil
from pathlib import Path


def pre_lama(args):
    in_dir = os.path.join(args.in_dir, f'{args.dataset_name}_sam', args.scene_name)
    out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_sam', args.scene_name)
    print(f'Prepare nerf dataset from {in_dir} to {out_dir}')

    # Read json params
    with open(args.pose_transform_json, 'r') as file:
        nerf_params = json.load(file)
        args.down_factor = nerf_params[args.dataset_name][args.scene_name]['down_factor']
        args.valid_views = nerf_params[args.dataset_name][args.scene_name]['valid_views']

        if isinstance(args.valid_views, int):
            args.int_views = [_ for _ in range(args.valid_views)]

    in_imgs_dir = os.path.join(in_dir, f'lama_out_refine')

    in_img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if
                    path.lower().endswith(args.img_file_type)]
    in_img_paths = sorted(in_img_paths)

    out_img_dir = os.path.join(out_dir, f'images_{args.down_factor}')
    os.makedirs(out_img_dir, exist_ok=True)

    ori_img_dir = os.path.join(in_dir, f'images_{args.down_factor}_ori')
    ori_img_paths = [os.path.join(ori_img_dir, path) for path in os.listdir(ori_img_dir) if
                     path.lower().endswith(args.img_file_type)]
    ori_img_paths = sorted(ori_img_paths)

    print(f'Copy imgs from {in_imgs_dir} to {out_img_dir}')
    for i, (in_img_path, ori_img_path) in enumerate(zip(in_img_paths, ori_img_paths)):
        filename = Path(ori_img_path).name
        out_img_path = os.path.join(out_img_dir, filename)

        # Copy img files directly
        shutil.copy(in_img_path, out_img_path)

    print(f'Reorganize pose file')
    in_pose_path = os.path.join(args.in_dir, f'{args.dataset_name}', args.scene_name, 'poses_bounds.npy')

    out_pose_path_ori = os.path.join(args.in_dir, f'{args.dataset_name}_sparse', args.scene_name, 'poses_bounds.npy')
    out_pose_path_sam = os.path.join(args.in_dir, f'{args.dataset_name}_sam', args.scene_name, 'poses_bounds.npy')

    # COLMAP pose is in chaos order, we need to specify this in training process
    # pose_matrix = np.load(in_pose_path)
    # pose_matrix_transform = pose_matrix[args.valid_views]
    # np.save(out_pose_path_ori, pose_matrix_transform)
    # np.save(out_pose_path_sam, pose_matrix_transform)
    shutil.copy(in_pose_path, out_pose_path_ori)
    shutil.copy(in_pose_path, out_pose_path_sam)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='', type=str, help='in data dir, containing images and masks')
    parser.add_argument('--out_dir', default='', type=str, help='lama input dir')
    parser.add_argument('--dataset_name', type=str, default='', help='')
    parser.add_argument('--scene_name', type=str, default='', help='')

    parser.add_argument('--pose_transform_json', default='', type=str, help='mask refine params json path')
    parser.add_argument('--down_factor', type=float, default=4, help='img resolution down scale factor')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


def main():
    args = parse()

    pre_lama(args)


if __name__ == '__main__':
    main()
