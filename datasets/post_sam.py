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
from datasets.mask_refine import mask_refine


def save_masks(masks, out_path):
    masks_img = (masks * 255).astype(np.uint8)
    cv2.imwrite(out_path, masks_img)


def draw_masks_on_img(img, masks, out_path):
    masks = masks.astype(np.int32)
    img[masks == 1] = 255

    cv2.imwrite(out_path, img)

    return img


def draw_points_on_img(img, points, labels, out_path, r=5):
    for coord, label in zip(points, labels):
        color = [0, 255, 0] if label == 1 else [0, 0, 255]
        cv2.circle(img, coord, r, color, -1)

    cv2.imwrite(out_path, img)

    return img


def filter_points2d_raw(points2d_raw, sort_indices, invalid_indices, num_points=1):
    points2d = list()
    for sort_index in sort_indices:
        sort_index_valid = sort_index.copy()

        if len(sort_index) > len(invalid_indices) > 0:
            sort_index_valid = np.setdiff1d(sort_index, invalid_indices, assume_unique=True)

        assert len(sort_index_valid) >= num_points, \
            f'valid projected points num {len(sort_index_valid)} < required sample points num {num_points}'

        points2d.append(points2d_raw[sort_index_valid[:num_points]])
    points2d = np.concatenate(points2d, axis=0)

    return points2d


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

        img0_points_pos = np.asarray(json_content[dataset_name][scene_name]['points'])
        img0_labels_pos = np.ones(len(img0_points_pos), dtype=np.int32)

        has_negative_prompt = False
        if 'points_negative' in json_content[dataset_name][scene_name]:
            has_negative_prompt = True

            img0_points_neg = np.asarray(json_content[dataset_name][scene_name]['points_negative'])
            img0_labels_neg = np.zeros(len(img0_points_neg), dtype=np.int32)

            img0_points = np.concatenate([img0_points_pos, img0_points_neg], axis=0)
            img0_labels = np.concatenate([img0_labels_pos, img0_labels_neg], axis=0)

            img0_labels_all_pos = np.ones_like(img0_labels)

        else:
            img0_points = img0_points_pos
            img0_labels = img0_labels_pos

        print(f'points prompt are: ')
        print(img0_points)
        print(img0_labels)

        down_factor = json_content[dataset_name][scene_name]['down_factor']
        num_points = json_content[dataset_name][scene_name]['num_points']

        is_mask_refine = False
        if 'refine_params' in json_content[dataset_name][scene_name]:
            refine_params = json_content[dataset_name][scene_name]['refine_params']
            is_mask_refine = True

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
    img_paths = sorted(img_paths)

    img0_path = img_paths[0]
    img0_filename = Path(img0_path).name
    img0_masks_path = os.path.join(out_masks_dir, img0_filename)
    img0_points_path = os.path.join(out_imgs_points_dir, img0_filename)
    img0_masked_path = os.path.join(out_imgs_masked_dir, img0_filename)

    # Draw points on the img
    img0 = cv2.imread(img0_path)
    draw_points_on_img(img0.copy(), img0_points, img0_labels, img0_points_path)

    print('Predict \'the first\' img by SAM...')
    masks0 = predict_by_sam_single_img(predictor, img0.copy(), img0_points, img0_labels)

    # May force to refine the mask area for involving negative points prompt if necessary
    if is_mask_refine:
        masks0 = mask_refine((masks0 * 255).astype(np.uint8), **refine_params)
        masks0[masks0 == 255] = 1

    save_masks(masks0.copy(), img0_masks_path)
    draw_masks_on_img(img0.copy(), masks0.copy(), img0_masked_path)

    masks_neg = None
    if has_negative_prompt:
        masks_all_pos = predict_by_sam_single_img(predictor, img0.copy(), img0_points, img0_labels_all_pos)
        masks_neg = masks_all_pos.copy()
        masks_neg[masks0 == 1] = 0

        # draw_masks_on_img(img0.copy(), masks_neg.copy(), img0_masked_path)

        # This neg mask also need to be refined
        refine_params_neg = refine_params['neg']
        masks_neg = mask_refine((masks_neg * 255).astype(np.uint8), **refine_params_neg)
        masks_neg[masks_neg == 255] = 1

        # draw_masks_on_img(img0.copy(), masks_neg.copy(), img0_masked_path)

    print('Done')
    print('--------------------------------')

    cam_dir = os.path.join(in_dir, 'sparse/0')
    cameras, images, points3D = read_model(path=cam_dir, ext='.bin')

    # Find img0's cam pose and depths
    print('Project the points prompt into 3d space according to COLMAP data structure')
    new_h, new_w = img0.shape[:2]
    img0_filestem = Path(img0_path).stem

    points3d, sort_indices = None, None
    points3d_pos, sort_indices_pos = None, None
    points3d_neg, sort_indices_neg = None, None

    for image_id, image in images.items():
        img_filestem = Path(image.name).stem
        if img_filestem == img0_filestem:
            assert img_filestem == img0_filestem

            K, R, t, w, h = gen_cam_param_colmap(cameras[image.camera_id], image)
            assert np.isclose(new_w / w, new_h / h)
            scale = new_w / w

            if not has_negative_prompt:
                points3d, sort_indices, _ = map_2d_to_3d_colmap(img0_points, masks0.copy(), image, points3D, scale)

                assert points3d is not None
                print(f'Find {len(points3d)} COLMAP feature points in the mask')

            # Uncomment this to debug
            # draw_points_on_img(img_inter.copy(), features, np.ones(len(features)), img0_points_path)

            else:
                points3d_pos, sort_indices_pos, _ = \
                    map_2d_to_3d_colmap(img0_points_pos, masks0.copy(), image, points3D, scale)
                points3d_neg, sort_indices_neg, _ = \
                    map_2d_to_3d_colmap(img0_points_neg, masks_neg.copy(), image, points3D, scale)

                assert points3d_pos is not None
                print(f'Find {len(points3d_pos)} COLMAP feature points in the pos mask')
                assert points3d_neg is not None
                print(f'Find {len(points3d_neg)} COLMAP feature points in the neg mask')

            break

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
                    assert img_filestem == Path(image.name).stem

                    cam = cameras[image.camera_id]

                    break

            # Project 3d points to 2d pixels
            img = cv2.imread(img_path)
            new_h, new_w = img.shape[:2]

            K, R, t, w, h = gen_cam_param_colmap(cam, image)
            assert np.isclose(new_w / w, new_h / h)

            if not has_negative_prompt:
                points2d_raw, invalid_indices = map_3d_to_2d_project(points3d, K, R, t, w, h, new_w, new_h)
                points2d = filter_points2d_raw(points2d_raw, sort_indices, invalid_indices, num_points)
                labels = np.ones((len(points2d)), dtype=np.int32)

            else:
                points2d_raw_pos, invalid_indices_pos = map_3d_to_2d_project(points3d_pos, K, R, t, w, h, new_w, new_h)
                points2d_raw_neg, invalid_indices_neg = map_3d_to_2d_project(points3d_neg, K, R, t, w, h, new_w, new_h)

                points2d_pos = filter_points2d_raw(points2d_raw_pos, sort_indices_pos, invalid_indices_pos, num_points)
                points2d_neg = filter_points2d_raw(points2d_raw_neg, sort_indices_neg, invalid_indices_neg, num_points)

                labels_pos = np.ones(len(points2d_pos), dtype=np.int32)
                labels_neg = np.zeros(len(points2d_neg), dtype=np.int32)

                points2d = np.concatenate([points2d_pos, points2d_neg], axis=0)
                labels = np.concatenate([labels_pos, labels_neg], axis=0)

            masks = predict_by_sam_single_img(predictor, img.copy(), points2d, labels)

            # May force to refine the mask area for involving negative points prompt if necessary
            if is_mask_refine:
                masks = mask_refine((masks * 255).astype(np.uint8), **refine_params)
                masks[masks == 255] = 1

            out_img_points_path = os.path.join(out_imgs_points_dir, img_filename)
            draw_points_on_img(img.copy(), points2d, labels, out_img_points_path)

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
