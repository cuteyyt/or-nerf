import json
import os
import sys

from pre_sam import parse

sys.path.append(os.getcwd())  # noqa
from utils.img import imgs2video


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    img_file_type = args.img_file_type

    # Read refine params from json path
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        down_factor = json_content[dataset_name][scene_name]['down_factor']

    # The sparse folder
    in_img_dir = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, f'images_{down_factor}')
    out_path = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name, 'input_views.mp4')
    # num_imgs = len([os.path.join(in_img_dir, path) for path in os.listdir(in_img_dir)
    #                 if path.lower().endswith(img_file_type)])
    # fps = num_imgs // 3
    imgs2video(in_img_dir, out_path, img_file_type, fps=10)

    # The sam folder
    in_img_dir = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, f'lama_out_refine')
    out_path = os.path.join(in_dir, f'{dataset_name}_sam', scene_name, 'lama.mp4')
    # num_imgs = len([os.path.join(in_img_dir, path) for path in os.listdir(in_img_dir)
    #                 if path.lower().endswith(img_file_type)])
    # fps = num_imgs // 3
    imgs2video(in_img_dir, out_path, img_file_type, fps=10)


if __name__ == '__main__':
    main()
