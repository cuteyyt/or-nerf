import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

sys.path.append("..")  # noqa
from colmap.read_write_model import read_model


def pre_lama(in_dir, out_dir):
    print(f'Pre-process for lama from {in_dir} to {out_dir}')

    rgb_dir = os.path.join(in_dir, 'imgs_ori')
    mask_dir = os.path.join(in_dir, 'masks')

    rgb_paths = [os.path.join(rgb_dir, path) for path in os.listdir(rgb_dir)]
    rgb_paths = sorted(rgb_paths)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    with tqdm(total=len(rgb_paths)) as t_bar:
        for i, rgb_path in enumerate(rgb_paths):
            filename = Path(rgb_path).name
            img_id = i

            mask_path = os.path.join(mask_dir, filename)

            out_img_path = os.path.join(out_dir, 'image{:0>3d}.png'.format(img_id))
            out_mask_path = os.path.join(out_dir, 'image{:0>3d}_mask{:0>3d}.png'.format(img_id, img_id))

            shutil.copy(rgb_path, out_img_path)

            img = cv2.imread(mask_path)
            img = cv2.dilate(img, dilate_kernel, iterations=3)
            cv2.imwrite(out_mask_path, img)
            # shutil.copy(mask_path, out_mask_path)

            t_bar.update(1)


# noinspection PyPep8Naming
def map_3d_to_2d(points3d, K, R, t, w, h):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # [R|t] transform XYZ_world to XYZ_cam
    # colmap pose: from world to camera
    pts_cam = np.matmul(R, points3d.transpose()) + t[:, np.newaxis]
    pts_cam = pts_cam.transpose()

    # get the depth value
    # depth_values = pts_cam[:, 2]

    # project the 3d points to 2d pixel coordinate
    # 2D normalized + multiply the intrinsic matrix (K)
    x_norm = pts_cam[:, 0] / pts_cam[:, 2]
    y_norm = pts_cam[:, 1] / pts_cam[:, 2]
    assert len(np.nonzero(pts_cam[:, 2] == 0)) != 0

    # new_h = 2268
    # new_w = 4032
    new_h = 2268 / 4.
    new_w = 4032 / 4.

    new_fx = fx * (new_w / w)
    new_fy = fy * (new_h / h)
    new_cx = cx * (new_w / w)
    new_cy = cy * (new_h / h)
    x_2d = x_norm * new_fx + new_cx
    y_2d = y_norm * new_fy + new_cy

    x_2d = np.round(x_2d).astype(np.int32)
    y_2d = np.round(y_2d).astype(np.int32)

    points2d = np.stack((x_2d, y_2d), axis=-1)
    return points2d


# noinspection PyPep8Naming
def gen_cam_pram(cam, img):
    camera_param = cam.params
    fx = camera_param[0]
    fy = camera_param[0]
    cx = camera_param[1]
    cy = camera_param[2]
    w = cam.width
    h = cam.height

    # assert cx == camera_param[2]
    # assert cy == camera_param[3]

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    R = img.qvec2rotmat()
    t = img.tvec

    return K, R, t, w, h


# noinspection PyPep8Naming
def map_2d_to_3d(masks, img_point, images, points3D, src_file_stem):
    # Find points in the img mask area, then find the points in 3d space corresponding to these points
    print(f'Find and map 2d points to 3d points for file_stem {src_file_stem}')
    tgt_image_id = 0
    for image_id, image in images.items():
        file_stem = Path(image.name).stem

        if file_stem == src_file_stem:
            tgt_image_id = image_id
            break

    tgt_image = images[tgt_image_id]
    points3d_ids_v = tgt_image.point3D_ids[tgt_image.point3D_ids > -1]

    points2d = list()
    points3d_indices = list()

    for point3d_idx in tqdm(points3d_ids_v):

        image_ids = points3D[point3d_idx].image_ids
        point2d_idxs = points3D[point3d_idx].point2D_idxs

        for image_id, point2d_idx in zip(image_ids, point2d_idxs):
            file_stem = Path(images[image_id].name).stem

            if file_stem == src_file_stem:
                point2d = images[image_id].xys[point2d_idx]
                point2d = point2d / 4.
                point2d = point2d.astype(np.int32)

                if masks[point2d[1], point2d[0]] == 1:
                    points2d.append(point2d)
                    points3d_indices.append(point3d_idx)

    assert len(points2d) > 0
    assert len(points3d_indices) > 0

    # Find the nearest feature point of the chosen point
    points2d = np.asarray(points2d)
    points3d_indices = np.asarray(points3d_indices)

    img_points = np.repeat(img_point, len(points2d), axis=0)
    points2d_dist = np.linalg.norm((points2d - img_points), axis=1)
    points_indices = np.argsort(points2d_dist)

    points2d = points2d[points_indices]
    points3d_indices = points3d_indices[points_indices]

    return points3d_indices, points2d


def draw_masks_on_img(img, masks, out_path):
    masks = masks.astype(np.int32)
    img[masks == 1] = 255

    cv2.imwrite(out_path, img)


def save_masks(masks, out_path):
    masks_img = (masks * 255).astype(np.uint8)
    cv2.imwrite(out_path, masks_img)


def draw_points_on_img(img, points, labels, out_path):
    if labels[0] == 1:
        color = [0, 0, 255]
    else:
        color = [0, 0, 0]

    for coord in points:
        cv2.circle(img, coord, 10, color, -1)

    cv2.imwrite(out_path, img)


def predict_by_sam_one_img(
        predictor,
        in_path,
        out_masks_path,
        out_img_points_path,
        out_img_masked_path,
        img_point,
        img_label):
    img = cv2.imread(in_path)

    draw_points_on_img(img.copy(), img_point, img_label, out_img_points_path)

    # Predict by sam
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=img_point,
        point_labels=img_label,
        multimask_output=True,
    )

    # Choose the confidence score >= 0.85
    confident_indices = scores >= 0.85

    # Choose the max-area mask for valid mask
    masks_valid = masks[confident_indices]
    assert len(masks_valid) > 0

    masks_imgs = masks_valid.astype(np.int32)
    masks_area = np.empty(len(masks_imgs))
    for i in range(len(masks_imgs)):
        mask_area = np.bincount(masks_imgs[i, :, :].reshape(-1))
        masks_area[i] = mask_area[1]
    target_masks = masks_valid[np.argmax(masks_area)]

    save_masks(target_masks.copy(), out_masks_path)
    draw_masks_on_img(img.copy(), target_masks.copy(), out_img_masked_path)

    target_masks = target_masks.astype(np.int32)
    return target_masks


def copy_imgs(in_dir, out_dir):
    print(f'Copy imgs from {in_dir} to {out_dir}')
    in_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if
                path.endswith('.jpg') or path.endswith('.png')]
    in_paths = sorted(in_paths)[40:]

    for in_path in tqdm(in_paths):
        filename = Path(in_path).name
        out_path = os.path.join(out_dir, filename)

        shutil.copy(in_path, out_path)


# noinspection PyPep8Naming
def post_sam(args, in_dir, out_dir):
    out_imgs_ori_dir = os.path.join(out_dir, 'imgs_ori')
    out_masks_dir = os.path.join(out_dir, 'masks')
    out_imgs_points_dir = os.path.join(out_dir, 'imgs_with_points')
    out_imgs_masked_dir = os.path.join(out_dir, 'imgs_with_masks')

    os.makedirs(out_imgs_ori_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    os.makedirs(out_imgs_points_dir, exist_ok=True)
    os.makedirs(out_imgs_masked_dir, exist_ok=True)

    # Copy imgs before inpainting
    in_imgs_dir = os.path.join(in_dir, 'images_4')
    copy_imgs(in_imgs_dir, out_imgs_ori_dir)

    # Register SAM model
    ckpt_path = args.ckpt_path
    model_type = args.model_type
    device = args.device_type

    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Predict the 'first' img

    # All img paths
    img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if
                 path.endswith('.jpg') or path.endswith('.png')]
    # TODO: pay attention to different filename
    img_paths = sorted(img_paths)[40:]

    img0_path = img_paths[0]
    img0_filename = Path(img0_path).name
    img0_masks_path = os.path.join(out_masks_dir, img0_filename)
    img0_points_path = os.path.join(out_imgs_points_dir, img0_filename)
    img0_masked_path = os.path.join(out_imgs_masked_dir, img0_filename)

    # TODO: point is magic number
    img0_point = np.array([[670, 190]])
    img0_label = np.array([1])

    print('Predict \'the first\' img by sam')
    masks0 = predict_by_sam_one_img(
        predictor, img0_path, img0_masks_path, img0_points_path, img0_masked_path, img0_point, img0_label)

    cam_dir = os.path.join(in_dir, 'sparse/0')
    cameras, images, points3D = read_model(path=cam_dir, ext='.bin')

    src_file_stem = Path(img0_path).stem
    img0_points3d_indices, _ = map_2d_to_3d(masks0, img0_point, images, points3D, src_file_stem)

    print('Predict other views by sam according to corresponding 2d points')
    with tqdm(total=len(img_paths) - 1) as t_bar:
        for i, img_path in enumerate(img_paths[1:]):
            img_filename = Path(img_path).name
            img_file_stem = Path(img_path).stem

            # Find the corresponding image id
            image_id = None
            for image_id, image in images.items():
                if img_file_stem == Path(image.name).stem:
                    break

            # Find the 2d points according to 3d points indices
            # One point is OK
            # More points may lead to bad results
            # Still need to do the projection
            points2d = np.zeros((2, 2), dtype=np.int32)
            cnt = 0
            for point3d_idx_ in img0_points3d_indices:
                image = images[image_id]
                camera = cameras[image.camera_id]
                K, R, t, w, h = gen_cam_pram(camera, image)
                point3d = points3D[point3d_idx_].xyz
                point2d = map_3d_to_2d(point3d[np.newaxis, :], K, R, t, w, h)

                if 0 <= point2d[0][0] < 1008 and 0 <= point2d[0][1] < 567:
                    points2d[cnt] = point2d
                    cnt += 1

                    if cnt >= 2:
                        break

            labels = np.ones((len(points2d)))

            out_masks_path = os.path.join(out_masks_dir, img_filename)
            out_img_points_path = os.path.join(out_imgs_points_dir, img_filename)
            out_img_masks_path = os.path.join(out_imgs_masked_dir, img_filename)
            predict_by_sam_one_img(
                predictor, img_path, out_masks_path, out_img_points_path, out_img_masks_path, points2d, labels)

            t_bar.update(1)


def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--in_dir', type=str, default='../../data/statue', help='todo')
    args.add_argument('--out_dir', type=str, default='../../data/statue-sam', help='todo')

    args.add_argument('--ckpt_path', type=str, default='../../ckpts/sam/sam_vit_h_4b8939.pth', help='todo')
    args.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h'], help='todo')
    args.add_argument('--device_type', type=str, default='cuda', choices=['cuda'], help='todo')

    opt = args.parse_args()
    return opt


# noinspection PyPep8Naming
def main():
    args = parse()
    in_dir = args.in_dir
    out_dir = args.out_dir

    # post_sam(args, in_dir, out_dir)

    out_lama_dir = os.path.join(out_dir, 'lama')
    os.makedirs(out_lama_dir, exist_ok=True)
    pre_lama(out_dir, out_lama_dir)

    print('Done')


if __name__ == '__main__':
    main()
