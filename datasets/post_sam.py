import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

sys.path.append(os.getcwd())  # noqa
from utils.cam import map_3d_to_2d_project, map_2d_to_3d_colmap, gen_cam_param_colmap
from utils.colmap.read_write_model import read_model


def save_masks(masks, out_path):
    masks_img = (masks * 255).astype(np.uint8)
    cv2.imwrite(out_path, masks_img)


def draw_masks_on_img(img, masks, out_path):
    masks = masks.astype(np.int32)
    img[masks == 1] = 255

    cv2.imwrite(out_path, img)


def draw_points_on_img(img, points, out_path, color=None, r=5):
    if color is None:
        color = [0, 0, 255]  # BGR, Red

    for coord in points:
        cv2.circle(img, coord, r, color, -1)

    cv2.imwrite(out_path, img)
    return img


def predict_by_sam_single_img(predictor, img, img_points, img_labels):
    # Predict by prepare_data
    predictor.set_image(img, image_format='BGR')
    masks, scores, logits = predictor.predict(
        point_coords=img_points,
        point_labels=img_labels,
        multimask_output=False,
    )
    masks_target = masks[0].astype(np.int32)

    return masks_target


def copy_imgs(in_dir, out_dir, img_file_type=None):
    print(f'Copy files from {in_dir} to {out_dir}')
    in_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.endswith(img_file_type)]

    for in_path in tqdm(in_paths):
        filename = Path(in_path).name
        out_path = os.path.join(out_dir, filename)

        shutil.copy(in_path, out_path)


# noinspection PyPep8Naming
def post_sam(args, ):
    # Args
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    img_file_type = args.img_file_type
    model_type, ckpt_path, device_type = args.model_type, args.ckpt_path, args.device_type

    # Read params from json
    print(f'Read params from json file {json_path}')
    with open(json_path, 'r') as file:
        json_content = json.load(file)

        img0_points = np.asarray(json_content[dataset_name][scene_name]['points'])
        img0_labels = np.ones(len(img0_points), dtype=np.int32)

        print(f'points prompt are: ')
        print(img0_points)

        down_factor = json_content[dataset_name][scene_name]['down_factor']
        num_points = json_content[dataset_name][scene_name]['num_points']
    print('--------------------------------')

    # Register related paths
    in_dir = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name)
    out_dir = os.path.join(out_dir, f'{dataset_name}_sam', scene_name)

    out_imgs_ori_dir = os.path.join(out_dir, f'images_{down_factor}_ori')
    out_masks_dir = os.path.join(out_dir, 'masks')
    out_imgs_points_dir = os.path.join(out_dir, 'imgs_with_points')
    out_imgs_masked_dir = os.path.join(out_dir, 'imgs_with_masks')

    os.makedirs(out_imgs_ori_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    os.makedirs(out_imgs_points_dir, exist_ok=True)
    os.makedirs(out_imgs_masked_dir, exist_ok=True)

    # Copy ori imgs from dense folder to prepare_data folder
    in_imgs_dir = os.path.join(in_dir, f'images_{down_factor}')
    copy_imgs(in_imgs_dir, out_imgs_ori_dir, img_file_type=img_file_type)
    print('--------------------------------')

    # Register SAM model
    print(f'Register SAM pretrained model {model_type} from {ckpt_path}...')
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device_type)
    predictor = SamPredictor(sam)
    print('Done')
    print('--------------------------------')

    # Predict the 'first' img
    img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if path.endswith(img_file_type)]
    # print(img_paths)
    img_paths = sorted(img_paths)

    img0_path = img_paths[0]
    img0_filename = Path(img0_path).name
    img0_masks_path = os.path.join(out_masks_dir, img0_filename)
    img0_points_path = os.path.join(out_imgs_points_dir, img0_filename)
    img0_masked_path = os.path.join(out_imgs_masked_dir, img0_filename)

    # Draw points on the img
    img0 = cv2.imread(img0_path)
    img_inter = draw_points_on_img(img0.copy(), img0_points, img0_points_path, color=[0, 0, 255])

    print('Predict \'the first\' img by SAM...')
    masks0 = predict_by_sam_single_img(predictor, img0.copy(), img0_points, img0_labels)
    save_masks(masks0.copy(), img0_masks_path)
    draw_masks_on_img(img0.copy(), masks0.copy(), img0_masked_path)
    print('Done')
    print('--------------------------------')

    cam_dir = os.path.join(in_dir, 'sparse/0')
    cameras, images, points3D = read_model(path=cam_dir, ext='.bin')

    # Find img0's cam pose and depths
    # TODO: What if there is no COLMAP feature points in the mask
    print('Project the points prompt into 3d space according to COLMAP data structure')
    new_h, new_w = img0.shape[:2]
    img0_filestem = Path(img0_path).stem
    points3d, sort_indices = None, None
    for image_id, image in images.items():
        img_filestem = Path(image.name).stem
        if img_filestem == img0_filestem:
            K, R, t, w, h = gen_cam_param_colmap(cameras[image.camera_id], image)
            assert np.isclose(new_w / w, new_h / h)
            scale = new_w / w

            points3d, sort_indices, features = map_2d_to_3d_colmap(img0_points, masks0.copy(), image, points3D, scale)
            draw_points_on_img(img_inter.copy(), features, img0_points_path, color=(255, 255, 255))

            break

    assert points3d is not None

    print(f'Find {len(points3d)} COLMAP feature points in the mask')
    print('--------------------------------')

    # Predict other views' mask by SAM
    print('Predict other views\' masks by SAM according to 2d-3d projection patchiness across views')
    with tqdm(total=len(img_paths) - 1) as t_bar:
        for i, img_path in enumerate(img_paths[1:]):
            img_filename = Path(img_path).name
            img_filestem = Path(img_path).stem

            # Find the corresponding image id
            cam, image = None, None
            for _, image in images.items():
                if img_filestem == Path(image.name).stem:
                    cam = cameras[image.camera_id]
                    break

            # Project 3d points to 2d pixels
            img = cv2.imread(img_path)
            new_h, new_w = img.shape[:2]

            K, R, t, w, h = gen_cam_param_colmap(cam, image)
            assert np.isclose(new_w / w, new_h / h)

            points2d_raw, invalid_indices = map_3d_to_2d_project(points3d, K, R, t, w, h, new_w, new_h)

            points2d = list()
            for sort_index in sort_indices:
                sort_index_valid = sort_index.copy()

                if len(invalid_indices) < len(sort_index):
                    sort_index_valid = np.setdiff1d(sort_index, invalid_indices, assume_unique=True)

                assert len(sort_index_valid) >= num_points, \
                    f'valid projected points num {len(sort_index_valid)} < required sample points num {num_points}'

                points2d.append(points2d_raw[sort_index_valid[:num_points]])
            points2d = np.concatenate(points2d, axis=0)

            out_img_points_path = os.path.join(out_imgs_points_dir, img_filename)
            draw_points_on_img(img.copy(), points2d, out_img_points_path)

            labels = np.ones((len(points2d)), dtype=np.int32)
            masks = predict_by_sam_single_img(predictor, img.copy(), points2d, labels)

            out_masks_path = os.path.join(out_masks_dir, img_filename)
            out_img_masks_path = os.path.join(out_imgs_masked_dir, img_filename)

            save_masks(masks.copy(), out_masks_path)
            draw_masks_on_img(img.copy(), masks.copy(), out_img_masks_path)

            t_bar.update(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--ckpt_path', type=str, help='prepare_data checkpoint path')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h'], help='prepare_data model type')
    parser.add_argument('--device_type', type=str, default='cuda', choices=['cuda'], help='device')

    parser.add_argument('--json_path', type=str)

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


# noinspection PyPep8Naming
def main():
    args = parse()

    post_sam(args)


if __name__ == '__main__':
    main()
