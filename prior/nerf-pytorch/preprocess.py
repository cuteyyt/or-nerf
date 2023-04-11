import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_files(in_dir, out_dir):
    in_img_dir = os.path.join(out_dir, 'lama_refine_out')
    out_img_dir = os.path.join(out_dir, 'images')
    os.makedirs(out_img_dir, exist_ok=True)

    print(f'Copy imgs from {in_dir} to {out_dir}')
    in_img_paths = [os.path.join(in_img_dir, path) for path in os.listdir(in_img_dir) if
                    path.endswith('.jpg') or path.endswith('.png')]

    for in_img_path in tqdm(in_img_paths):
        # filename = Path(in_img_path).name
        filename_id = int(Path(in_img_path).stem.split('_')[1][4:])

        out_img_path = os.path.join(out_img_dir, '{:0>3d}.png'.format(filename_id))

        shutil.copy(in_img_path, out_img_path)

    print(f'Copy cam params files from {in_dir} to {out_dir}')
    shutil.copy(os.path.join(in_dir, 'poses_bounds.npy'), os.path.join(out_dir, 'poses_bounds.npy'))


def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--in_dir', type=str, default='../../data/statue', help='')
    args.add_argument('--out_dir', type=str, default='../../data/statue-sam', help='')

    opt = args.parse_args()
    return opt


def main():
    args = parse()
    in_dir = args.in_dir
    out_dir = args.out_dir

    copy_files(in_dir, out_dir)


if __name__ == '__main__':
    main()
