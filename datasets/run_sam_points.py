"""
Prepare a 'sam' folder for running scenes with delete and lama-processing
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
from scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

from utils.cam import map_3d_to_2d_project, map_2d_to_3d_colmap, gen_cam_param_colmap
from utils.colmap.read_write_model import read_model
from utils.mask_refine import mask_refine


def save_masks(masks, out_path):
    masks_img = (masks * 255).astype(np.uint8)
    cv2.imwrite(out_path, masks_img)

    return masks_img


def draw_masks_on_img(img, masks, out_path):
    img[masks == 1] = 255
    cv2.imwrite(out_path, img)

    return img


def draw_points_on_img(img, points, labels, out_path, r=5):
    for coord, label in zip(points, labels):
        color = [0, 255, 0] if label == 1 else [0, 0, 255]  # green if label==1 else red
        cv2.circle(img, coord, r, color, -1)

    cv2.imwrite(out_path, img)

    return img


def filter_points2d(points2d_raw, sort_indices, invalid_indices, num_points=1):
    points2d = list()
    for sort_index in sort_indices:
        sort_index_valid = np.copy(sort_index)

        if 0 < len(invalid_indices) < len(sort_index):
            sort_index_valid = np.setdiff1d(sort_index, invalid_indices, assume_unique=True)

        assert len(sort_index_valid) >= num_points, \
            f'valid projected points num {len(sort_index_valid)} < required sample points num {num_points}'

        points2d.append(points2d_raw[sort_index_valid[:num_points]])
    points2d = np.concatenate(points2d, axis=0)

    return points2d


# noinspection PyPep8Naming
def pred_all_views(predictor, img_paths, out_dirs, points, masks, cameras, images, points3D, opt):
    print('Predict all views\' masks by SAM according to 2d-3d projection patchiness across views')
    num_points = opt['num_points']

    pred_idx = 0
    if 'pred_params' in opt:
        if 'pred_idx' in opt['pred_params']:
            pred_idx = opt['pred_params']['pred_idx']

    img_paths.insert(0, img_paths[pred_idx])
    img_paths.pop(pred_idx + 1)

    dynamic_param = dict()
    with tqdm(total=len(img_paths)) as t_bar:
        for i, img_path in enumerate(img_paths):
            img_filestem = Path(img_path).stem

            # Find the corresponding image id
            cam, image = None, None
            for _, image in images.items():
                if Path(image.name).stem == img_filestem:
                    assert Path(image.name).stem == img_filestem
                    cam = cameras[image.camera_id]

                    break

            # Project 2d points to 3d pixels (Find 3d space)
            img = cv2.imread(img_path)

            if 'progressive' in opt:  # Use the previous pred mask to update the 3d space
                if not opt['has_neg_prompt']:
                    points_ = {'p': dynamic_param['points']}
                else:
                    points_ = {'p': dynamic_param['points'],
                               'n': dynamic_param['points_n']}
                masks_ = dynamic_param['masks']
                res_3d = find_3d_space(img_path, img, points_, masks_, cameras, images, points3D, opt)
            else:
                if i == pred_idx:  # Only need for the first image
                    if not opt['has_neg_prompt']:
                        points_ = {'p': points}
                    else:
                        points_ = {'p': opt['init_pts_p'],
                                   'n': opt['init_pts_n']}
                    masks_ = masks
                    res_3d = find_3d_space(img_paths[0], img, points_, masks_, cameras, images, points3D, opt)

            # Project 3d points to 2d pixels
            new_h, new_w = img.shape[:2]
            K, R, t, w, h = gen_cam_param_colmap(cam, image)

            assert np.isclose(new_w / w, new_h / h)

            if not opt['has_neg_prompt']:
                points2d_raw, invalid_indices = map_3d_to_2d_project(res_3d['points3d'], K, R, t, w, h, new_w, new_h)
                points2d = filter_points2d(points2d_raw, res_3d['sort_indices'], invalid_indices, num_points)
                labels = np.ones((len(points2d)), dtype=np.int32)

            else:
                pts2d_raw_p, invalid_i_p = map_3d_to_2d_project(res_3d['p']['points3d'], K, R, t, w, h, new_w, new_h)
                pts2d_raw_n, invalid_i_n = map_3d_to_2d_project(res_3d['n']['points3d'], K, R, t, w, h, new_w, new_h)

                points2d_pos = filter_points2d(pts2d_raw_p, res_3d['p']['sort_indices'], invalid_i_p, num_points)
                points2d_neg = filter_points2d(pts2d_raw_n, res_3d['n']['sort_indices'], invalid_i_n, num_points)

                labels_pos = np.ones(len(points2d_pos), dtype=np.int32)
                labels_neg = np.zeros(len(points2d_neg), dtype=np.int32)

                points2d = np.concatenate([points2d_pos, points2d_neg], axis=0)
                labels = np.concatenate([labels_pos, labels_neg], axis=0)

            masks2d = pred_warp(predictor, img_path, out_dirs, img, points2d, labels, opt)

            if not opt['has_neg_prompt']:
                dynamic_param = {
                    'points': points2d,
                    'masks': masks2d,
                }
            else:
                dynamic_param = {
                    'points': points2d_pos,
                    'points_n': points2d_neg,
                    'masks': masks2d,
                }

            t_bar.update(1)


# noinspection PyPep8Naming,GrazieInspection
def find_3d_space(img_path, img, points, masks, cameras, images, points3D, opt):
    # Find img0's cam pose and depths
    # print('Project the points prompt into 3d space according to COLMAP data structure')

    new_h, new_w = img.shape[:2]
    src_img_filestem = Path(img_path).stem

    res = dict()

    for image_id, image in images.items():
        img_filestem = Path(image.name).stem
        if img_filestem == src_img_filestem:
            assert img_filestem == src_img_filestem

            K, R, t, w, h = gen_cam_param_colmap(cameras[image.camera_id], image)

            assert np.isclose(new_w / w, new_h / h)
            scale = new_w / w

            if not opt['has_neg_prompt']:
                pts3d, sort_i, ft = map_2d_to_3d_colmap(points['p'], masks['masks'], image, points3D, scale)

                assert pts3d is not None
                # print(f'Find {len(pts3d)} COLMAP feature points in the mask')

                # draw_points_on_img(img.copy(), ft, np.ones(len(ft)), img_points_path)

                res = {
                    'points3d': pts3d,
                    'sort_indices': sort_i,
                    'feature_points': ft
                }

            else:
                pts3d_p, sort_i_p, ft_p = map_2d_to_3d_colmap(points['p'], masks['masks'], image, points3D, scale)
                pts3d_n, sort_i_n, ft_n = map_2d_to_3d_colmap(points['n'], masks['masks_n'], image, points3D, scale)

                assert pts3d_p is not None
                # print(f'Find {len(pts3d_p)} COLMAP feature points in the pos mask')
                assert pts3d_n is not None
                # print(f'Find {len(pts3d_n)} COLMAP feature points in the neg mask')

                res = {
                    'p': {'points3d': pts3d_p,
                          'sort_indices': sort_i_p,
                          'feature_points': ft_p},
                    'n': {'points3d': pts3d_n,
                          'sort_indices': sort_i_n,
                          'feature_points': ft_n}
                }

            break

    # print('--------------------------------')

    return res


def choose_mask(masks, scores, logits, threshold):
    masks_valid = masks[scores >= threshold]
    assert len(masks_valid) >= 1

    scores_valid = scores[scores >= threshold]
    logits_valid = logits[scores >= threshold]

    masks_flat = masks.reshape(masks_valid.shape[0], -1)

    masks_areas = np.sum(masks_flat, axis=1)
    sort_indices = np.argsort(masks_areas)

    mask = masks_valid[sort_indices][-1][np.newaxis, ...]
    score = scores_valid[sort_indices][-1]
    logit = logits_valid[sort_indices][-1][np.newaxis, ...]

    return mask, score, logit


def predict_by_sam_single_img(predictor, img, img_points, img_labels, params):
    predictor.set_image(img, image_format='BGR')

    masks, scores, logits = None, None, None
    if params['has_pred_params']:
        for _ in range(params['pred_params']['pred_iters']):
            if 'threshold' in params['pred_params']:
                multimask_output_flag = True
                threshold = params['pred_params']['threshold']
            else:
                multimask_output_flag = False
                threshold = None

            if logits is None:
                masks, scores, logits = predictor.predict(
                    point_coords=img_points,
                    point_labels=img_labels,
                    multimask_output=multimask_output_flag,
                    mask_input=None
                )
                if multimask_output_flag:
                    masks, scores, logits = choose_mask(masks, scores, logits, threshold)
            else:
                masks, scores, logits = predictor.predict(
                    point_coords=img_points,
                    point_labels=img_labels,
                    multimask_output=multimask_output_flag,
                    mask_input=logits
                )
                if multimask_output_flag:
                    masks, scores, logits = choose_mask(masks, scores, logits, threshold)

    else:
        masks, scores, logits = predictor.predict(
            point_coords=img_points,
            point_labels=img_labels,
            multimask_output=False,

        )

    masks_target = masks[0].astype(np.int32)
    return masks_target


def pred_warp(predictor, img_path, out_dirs, img, points, labels, opt):
    # print('Predict \'the first\' img by SAM...')

    img_filename = Path(img_path).name

    img_masks_path = os.path.join(out_dirs['masks_dir'], img_filename)
    img_points_path = os.path.join(out_dirs['imgs_points_dir'], img_filename)
    img_masked_path = os.path.join(out_dirs['imgs_masked_dir'], img_filename)

    draw_points_on_img(img.copy(), points, labels, img_points_path)

    masks = predict_by_sam_single_img(predictor, img, points, labels, opt)

    # May force to refine the mask area for involving negative points prompt if necessary
    if opt['has_mask_params']:
        masks_img = (masks * 255).astype(np.uint8)
        masks_img = mask_refine(masks_img, opt['refine_params'])
        masks[masks_img == 255] = 1

    save_masks(masks.copy(), img_masks_path)
    draw_masks_on_img(img.copy(), masks.copy(), img_masked_path)

    res = {'masks': masks}

    if opt['has_neg_prompt']:
        masks_all_p = predict_by_sam_single_img(predictor, img, points, opt['init_lbs_all_p'], opt)

        masks_neg = np.copy(masks_all_p)
        masks_neg[masks == 1] = 0
        # draw_masks_on_img(img.copy(), masks_neg.copy(), img_masked_path)

        # This neg mask also need to be refined
        if 'neg' in opt['refine_params']:
            refine_params_neg = opt['refine_params']['neg']
            masks_neg_img = (masks_neg * 255).astype(np.uint8)

            masks_neg_img = mask_refine(masks_neg_img, refine_params_neg)
            masks_neg[masks_neg_img == 255] = 1
            # draw_masks_on_img(img.copy(), masks_neg.copy(), img_masked_path)

        res['masks_n'] = masks_neg

    # print('Done')
    # print('--------------------------------')

    return res


def load_sam(model_type, ckpt_path, device_type):
    print(f'Register SAM pretrained model {model_type} from {ckpt_path}...')
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device_type)
    predictor = SamPredictor(sam)
    print('Done')
    print('--------------------------------')
    return predictor


def read_json(json_path, dataset_name, scene_name):
    print(f'Read params from json file {json_path}')
    with open(json_path, 'r') as file:
        json_content = json.load(file)
        scene_params = json_content[dataset_name][scene_name]

        res = {
            'down_factor': scene_params['down_factor'],
            'num_points': scene_params['num_points'],
            'mode': scene_params['mode'],
            'has_neg_prompt': False,
            'has_mask_params': False,
            'has_pred_params': False
        }

        init_pts_p = np.asarray(scene_params['points'])
        init_lbs_p = np.ones(len(init_pts_p), dtype=np.int32)

        if 'points_negative' in scene_params:
            init_pts_n = np.asarray(scene_params['points_negative'])
            init_lbs_n = np.zeros(len(init_pts_n), dtype=np.int32)

            init_points = np.concatenate([init_pts_p, init_pts_n], axis=0)
            init_labels = np.concatenate([init_lbs_p, init_lbs_n], axis=0)

            init_lbs_all_p = np.ones_like(init_labels)

            res['has_neg_prompt'] = True
            res['init_lbs_all_p'] = init_lbs_all_p

            res['init_pts_p'] = init_pts_p
            res['init_pts_n'] = init_pts_n

        else:
            init_points = init_pts_p
            init_labels = init_lbs_p

        print(f'points prompt are: ')
        print(init_points)
        print(init_labels)

        res['init_points'] = init_points
        res['init_labels'] = init_labels

        if 'refine_params' in scene_params:
            res['has_mask_params'] = True
            res['refine_params'] = scene_params['refine_params']

        if 'pred_params' in scene_params:
            res['has_pred_params'] = True
            res['pred_params'] = scene_params['pred_params']

    print('--------------------------------')

    return res


def sample_points_from_mask(mask, mode='use_centre'):
    box = np.nonzero(mask[:, :, 0])
    print(box)

    mask_binary = np.zeros(mask.shape[:2], dtype=np.int32)
    mask_binary[mask[:, :, 0] == 255] = 1

    centre = ndimage.center_of_mass(mask_binary)

    if mode == 'use_all':
        points = [
            [box[1][0], box[0][0]],
            [int(centre[1]), int(centre[0])],
            [box[1][-1], box[0][-1]],
        ]
    elif mode == 'use_edge':
        points = [
            [box[1][0], box[0][0]],
            # [int(centre[1]), int(centre[0])],
            [box[1][-1], box[0][-1]],
        ]
    elif mode == 'use_centre':
        points = [
            # [box[1][0], box[0][0]],
            [int(centre[1]), int(centre[0])],
            # [box[1][-1], box[0][-1]],
        ]
    else:
        raise ValueError(f'Not implemented mode {mode}')

    points = np.asarray(points)
    return points


# noinspection PyPep8Naming
def post_sam(args, ):
    # Args
    in_dir, out_dir = args.in_dir, args.out_dir
    dataset_name, scene_name = args.dataset_name, args.scene_name
    json_path = args.json_path
    model_type, ckpt_path, device_type = args.model_type, args.ckpt_path, args.device_type
    text_prompt, is_test = args.text_prompt, args.is_test

    # Read params from json
    opt = read_json(json_path, dataset_name, scene_name)
    if args.text_prompt:
        text_mask_dir = os.path.join(out_dir, f'{dataset_name}_sam_text', f'{scene_name}', 'masks')
        text_mask_path = sorted([os.path.join(text_mask_dir, f) for f in os.listdir(text_mask_dir)])[0]
        mask = cv2.imread(text_mask_path)
        mode = opt['mode']
        opt['init_points'] = sample_points_from_mask(mask, mode)
        opt['init_labels'] = np.ones(len(opt['init_points']), dtype=np.int32)

    # Register related paths
    in_dir = os.path.join(in_dir, f'{dataset_name}_sparse', scene_name)
    in_imgs_dir = os.path.join(in_dir, f'images_{opt["down_factor"]}')

    in_cam_dir = os.path.join(in_dir, 'sparse/0')

    # Copy cam params to sam folder
    if text_prompt:
        out_dir = os.path.join(out_dir, f'{dataset_name}_sam_text', scene_name)
    else:
        out_dir = os.path.join(out_dir, f'{dataset_name}_sam', scene_name)
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(os.path.join(in_dir, 'poses_bounds.npy'), os.path.join(out_dir, 'poses_bounds.npy'))

    out_cam_dir = os.path.join(out_dir, 'sparse/0')
    os.makedirs(out_cam_dir, exist_ok=True)

    for file in os.listdir(in_cam_dir):
        shutil.copy(os.path.join(in_cam_dir, file), os.path.join(out_cam_dir, file))

    # Read COLMAP cam params
    cameras, images, points3D = read_model(path=out_cam_dir, ext='.bin')

    img_names = [images[k].name for k in images]
    img_names = np.sort(img_names)

    if is_test:
        img_names = img_names[40:]

    # img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if path.endswith(img_file_type)]
    img_paths = [os.path.join(in_imgs_dir, f) for f in img_names]

    out_dirs = {
        'imgs_dir': os.path.join(out_dir, 'images'),
        'masks_dir': os.path.join(out_dir, 'masks'),
        'imgs_ori_dir': os.path.join(out_dir, f'images_{opt["down_factor"]}_ori'),
        'imgs_points_dir': os.path.join(out_dir, 'imgs_with_points'),
        'imgs_masked_dir': os.path.join(out_dir, 'imgs_with_masks'),
    }

    for key, item in out_dirs.items():
        os.makedirs(item, exist_ok=True)

    # Copy ori imgs from dense folder to prepare_data folder
    check_output('cp {}/*.png {}'.format(in_imgs_dir, out_dirs['imgs_ori_dir']), shell=True)
    check_output('cp {}/*.png {}'.format(os.path.join(in_dir, 'images'), out_dirs['imgs_dir']), shell=True)

    # Register SAM model
    predictor = load_sam(model_type, ckpt_path, device_type)

    # Predict the 'first' img
    points = opt['init_points']
    labels = opt['init_labels']

    img = cv2.imread(img_paths[0])
    masks = pred_warp(predictor, img_paths[0], out_dirs, img, points, labels, opt)

    # Predict other views' mask by SAM
    pred_all_views(predictor, img_paths, out_dirs, points, masks, cameras, images, points3D, opt)
    print('--------------------------------')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--scene_name', type=str)

    parser.add_argument('--ckpt_path', type=str, help='prepare_data checkpoint path')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h'], help='prepare_data model type')
    parser.add_argument('--device_type', type=str, default='cuda', choices=['cuda'], help='device')

    parser.add_argument('--json_path', type=str, default='configs/prepare_data/sam_points.json')

    parser.add_argument('--text_prompt', action='store_true')
    parser.add_argument('--is_test', action='store_true')

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


# noinspection PyPep8Naming
def main():
    args = parse()

    post_sam(args)


if __name__ == '__main__':
    main()
