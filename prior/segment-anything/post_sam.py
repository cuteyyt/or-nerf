import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

sys.path.append(os.getcwd())  # noqa

from utils.colmap.read_write_model import read_model
from utils.file import copy_files


# noinspection PyPep8Naming
def map_3d_to_2d_by_project(points3d, K, R, t, w, h, new_w, new_h):
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

    new_fx = fx * (new_w / w)
    new_fy = fy * (new_h / h)
    new_cx = cx * (new_w / w)
    new_cy = cy * (new_h / h)
    x_2d = x_norm * new_fx + new_cx
    y_2d = y_norm * new_fy + new_cy

    x_2d = np.round(x_2d).astype(np.int32)
    y_2d = np.round(y_2d).astype(np.int32)
    points2d = np.array([x_2d, y_2d]).transpose()

    invalid_indices = list()
    for i, (x, y) in enumerate(zip(x_2d, y_2d)):
        if (x < 0) or (y < 0) or (x >= new_w) or (y >= new_h):
            continue
        else:
            invalid_indices.append(i)

    invalid_indices = np.asarray(invalid_indices, dtype=np.int32)

    return points2d, invalid_indices


# noinspection PyPep8Naming
def map_2d_to_3d_by_project(points2d, K, R, t, depths, scale=1.):
    # points2d : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : Translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.

    R = R.T
    t = -R @ t
    t.shape = (3, 1)

    points3d = list()
    for i, (p, d) in enumerate(zip(points2d, depths)):
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T
        p.shape = (3, 1)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p

        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)

        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0]).T
        cam.shape = (3, 1)
        cam_world = t + R @ cam

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)

        # Point scaled along this ray
        p3d = cam_world + scale * d * unit_vector

        points3d.append(p3d.squeeze(1))

    return np.asarray(points3d)


# noinspection PyPep8Naming
def map_2d_to_3d_by_colmap(points2d, masks, image, points3D, scale=1.):
    points3d_indices_for_img = image.point3D_ids

    points3d = list()
    pixel_coords = list()
    # print(image)
    for i, coord in enumerate(image.xys):
        # print(coord)
        point2d_scale = (coord * scale).astype(np.int32)
        # print(points3d_indices_for_img[i])
        # print(masks[point2d_scale[1], point2d_scale[0]])
        if points3d_indices_for_img[i] > -1 and masks[point2d_scale[1], point2d_scale[0]] == 1:
            points3d.append(points3D[points3d_indices_for_img[i]].xyz)
            pixel_coords.append(point2d_scale)

    points3d = np.asarray(points3d)
    pixel_coords = np.asarray(pixel_coords)

    # Find mask contour
    # masks_img = np.copy(masks).astype(np.uint8)
    # masks_img[masks == 1] = 255
    # contour = find_contours(masks_img)

    dists = np.empty((len(points2d), len(points3d)), dtype=np.float64)
    for i, point2d in enumerate(points2d):
        dists[i] = np.linalg.norm((point2d - pixel_coords), axis=1)
    sort_indices = np.argsort(dists, axis=1)
    # print(points3d.shape, sort_indices.shape)

    return points3d, sort_indices, pixel_coords


# noinspection PyPep8Naming
def gen_cam_pram(cam, img):
    camera_param = cam.params

    if cam.model == 'PINHOLE':
        fx = camera_param[0]
        fy = camera_param[1]
        cx = camera_param[2]
        cy = camera_param[3]
        w = cam.width
        h = cam.height
        # print(fx, fy)

        # assert fx == fy

    elif cam.model == 'SIMPLE_RADIAL':
        fx = camera_param[0]
        fy = camera_param[0]
        cx = camera_param[1]
        cy = camera_param[2]
        w = cam.width
        h = cam.height

    else:
        raise RuntimeError

    assert cx == w / 2.
    assert cy == h / 2.

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    R = img.qvec2rotmat()
    t = img.tvec

    return K, R, t, w, h


def find_contours(img, ):
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    return contours[0]


def draw_pcd(points, colors, out_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)


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


def predict_by_sam_single_img(predictor, img, img_points, img_labels, confidence_score=0.85):
    # Predict by sam
    predictor.set_image(img, image_format='BGR')
    # masks, scores, logits = predictor.predict(
    #     point_coords=img_points,
    #     point_labels=img_labels,
    #     multimask_output=True,
    # )
    #
    # # Choose the confidence score >= 0.85
    # confident_indices = scores >= confidence_score
    #
    # # Choose the max-area mask for valid mask
    # masks_valid = masks[confident_indices]
    # assert len(masks_valid) > 0
    #
    # masks_imgs = masks_valid.astype(np.int32)
    # masks_area = np.empty(len(masks_imgs))
    # for i in range(len(masks_imgs)):
    #     mask_area = np.bincount(masks_imgs[i, :, :].reshape(-1))
    #     masks_area[i] = mask_area[1]
    #
    # masks_target = masks_valid[np.argmax(masks_area)]
    # masks_target = masks_target.astype(np.int32)

    masks, scores, logits = predictor.predict(
        point_coords=img_points,
        point_labels=img_labels,
        multimask_output=False,
    )
    masks_target = masks[0].astype(np.int32)

    return masks_target


def convert_llff_filename(images, img_paths):
    # Solving conflict between existing images filename and read from image.bin filename
    res = {}
    images_read = []
    images_exist = []
    for i in range(len(img_paths)):
        img_i = img_paths[i]
        img_i_filename = Path(img_i).stem
        images_exist.append(img_i_filename)
    for image_id, image in images.items():
        image_filestem = Path(image.name).stem
        images_read.append(image_filestem)
    images_read.sort()
    images_exist.sort()

    # Special judgement for spinnerf dataset
    if (len(images_read) - len(images_exist)) == 40:
        images_exist = ["" for i in range(40)] + images_exist

    assert len(images_read) == len(images_exist)
    for i in range(len(images_read)):
        res[images_read[i]] = images_exist[i]
    return res


# noinspection PyPep8Naming
def post_sam(args, ):
    # Read init points prompt
    print(f'Read click-points prompt and other related params from json file {args.points_json_path}')
    with open(args.points_json_path, 'r') as file:
        sam_params = json.load(file)

        img0_points = np.asarray(sam_params[args.dataset_name][args.scene_name]['points'])
        img0_labels = np.ones(len(img0_points), dtype=np.int32)

        print(f'points prompt are: ')
        print(img0_points)

        # Other predict params
        args.down_factor = sam_params[args.dataset_name][args.scene_name]['down_factor']
        args.num_points = sam_params[args.dataset_name][args.scene_name]['num_points']
        args.confidence_score = sam_params[args.dataset_name][args.scene_name]['confidence_score']
    print('--------------------------------')

    # Register related paths
    in_dir = os.path.join(args.in_dir, f'{args.dataset_name}_sparse', args.scene_name)
    out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_sam', args.scene_name)

    out_imgs_ori_dir = os.path.join(out_dir, f'images_{args.down_factor}_ori')
    out_masks_dir = os.path.join(out_dir, 'masks')
    out_imgs_points_dir = os.path.join(out_dir, 'imgs_with_points')
    out_imgs_masked_dir = os.path.join(out_dir, 'imgs_with_masks')

    os.makedirs(out_imgs_ori_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    os.makedirs(out_imgs_points_dir, exist_ok=True)
    os.makedirs(out_imgs_masked_dir, exist_ok=True)

    # Copy ori imgs from dense folder to sam folder
    in_imgs_dir = os.path.join(in_dir, f'images_{args.down_factor}')
    copy_files(in_imgs_dir, out_imgs_ori_dir, file_type=args.img_file_type)
    print('--------------------------------')

    # Register SAM model
    print(f'Register SAM pretrained model {args.model_type} from {args.ckpt_path}...')
    ckpt_path = args.ckpt_path
    model_type = args.model_type
    device = args.device_type

    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print('Done')
    print('--------------------------------')

    # Predict the 'first' img
    img_paths = [os.path.join(in_imgs_dir, path) for path in os.listdir(in_imgs_dir) if
                 path.lower().endswith(args.img_file_type)]
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
    masks0 = predict_by_sam_single_img(predictor, img0, img0_points, img0_labels, args.confidence_score)
    save_masks(masks0.copy(), img0_masks_path)
    draw_masks_on_img(img0.copy(), masks0.copy(), img0_masked_path)
    print('Done')
    print('--------------------------------')

    cam_dir = os.path.join(in_dir, 'sparse/0')
    cameras, images, points3D = read_model(path=cam_dir, ext='.bin')

    convert_dict = convert_llff_filename(images, img_paths)

    # Find img0's cam pose and depths
    # TODO: What if there is no COLMAP feature points in the mask
    print('Project the points prompt into 3d space according to COLMAP data structure')
    new_h, new_w = img0.shape[:2]
    img0_filestem = Path(img0_path).stem
    points3d, sort_indices = None, None
    for image_id, image in images.items():
        image_filestem = Path(image.name).stem
        # print(image_filestem)
        # print(img0_filestem)
        if convert_dict[image_filestem] == img0_filestem:
            K, R, t, w, h = gen_cam_pram(cameras[image.camera_id], image)
            scale = new_w / w
            # print(scale, new_h / h)
            assert np.abs(new_w / w - new_h / h) < 1e-2

            # points2d_scale = (img0_points * scale).astype(np.int32)
            # # img0_mask_points = np.transpose(np.nonzero(masks0))
            # # points2d_scale = (img0_mask_points * scale).astype(np.int32)
            #
            # depth_path = os.path.join(in_dir, 'dense/0/stereo/depth_maps/', f'{image_filename}.photometric.bin')
            # depth_map = read_array(depth_path)
            #
            # min_depth, max_depth = np.percentile(depth_map, [args.min_depth_percentile, args.max_depth_percentile])
            # depth_map[depth_map < min_depth] = min_depth
            # depth_map[depth_map > max_depth] = max_depth
            #
            # depths = depth_map[(points2d_scale[:, 0], points2d_scale[:, 1])]
            # points3d = map_2d_to_3d_by_project(points2d_scale[:, -1::-1], K, R, t, depths, scale=scale)

            # masks0_tmp = np.zeros(img0.shape[:2], dtype=np.int32)
            # masks0_tmp[(points2d_scale[:, 0], points2d_scale[:, 1])] = 1
            # draw_masks_on_img(img0.copy(), masks0_tmp.copy(), img0_points_path)
            # draw_pcd(points3d, np.ones_like(points3d), os.path.join(out_dir, 'debug.pcd'))

            points3d, sort_indices, pixel_coords = map_2d_to_3d_by_colmap(img0_points, masks0, image, points3D, scale)

            draw_points_on_img(img_inter.copy(), pixel_coords, img0_points_path, color=(255, 255, 255))

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
                if img_filestem == convert_dict[Path(image.name).stem]:
                    cam = cameras[image.camera_id]
                    break

            # Project 3d points to 2d pixels
            img = cv2.imread(img_path)
            new_h, new_w = img.shape[:2]
            K, R, t, w, h = gen_cam_pram(cam, image)

            # assert np.abs(new_w / w - new_h / h) < 1e-2
            assert np.isclose(new_w / w, new_h / h)

            points2d_raw, invalid_indices = map_3d_to_2d_by_project(points3d, K, R, t, w, h, new_w, new_h)

            points2d = list()
            for sort_index in sort_indices:
                sort_index_valid = sort_index.copy()
                if len(invalid_indices) < len(sort_index):
                    sort_index_valid = np.setdiff1d(sort_index, invalid_indices, assume_unique=True)
                # print(sort_index_valid)

                assert len(sort_index_valid) >= args.num_points, \
                    f'valid projected points num {len(sort_index_valid)} < required sample points num {args.num_points}'

                points2d.append(points2d_raw[sort_index_valid[:args.num_points]])
            points2d = np.concatenate(points2d, axis=0)

            # mask = np.zeros(img.shape[:2], dtype=np.int32)
            # mask[(points2d[:, 0], points2d[:, 1])] = 1
            # out_img_points_path = os.path.join(out_imgs_points_dir, img_filename)
            # draw_masks_on_img(img.copy(), mask.copy(), out_img_points_path)
            # continue

            out_img_points_path = os.path.join(out_imgs_points_dir, img_filename)
            draw_points_on_img(img.copy(), points2d.copy(), out_img_points_path)

            labels = np.ones((len(points2d)), dtype=np.int32)
            masks = predict_by_sam_single_img(predictor, img, points2d, labels)

            out_masks_path = os.path.join(out_masks_dir, img_filename)
            out_img_masks_path = os.path.join(out_imgs_masked_dir, img_filename)

            save_masks(masks.copy(), out_masks_path)
            draw_masks_on_img(img.copy(), masks.copy(), out_img_masks_path)

            t_bar.update(1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='', help='')
    parser.add_argument('--out_dir', type=str, default='', help='')
    parser.add_argument('--dataset_name', type=str, default='', help='')
    parser.add_argument('--scene_name', type=str, default='', help='')

    parser.add_argument('--ckpt_path', type=str, default='', help='sam checkpoint path')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h'], help='sam model type')

    parser.add_argument('--device_type', type=str, default='cuda', choices=['cuda'], help='device')

    parser.add_argument('--points_json_path', type=str, default='', help='init points loc json path')
    parser.add_argument('--down_factor', type=float, default=4, help='img resolution down scale factor')
    parser.add_argument('--confidence_score', type=float, default=0.85, help='sam output confidence threshold')
    parser.add_argument("--num_points", default=3, type=int,
                        help='sample num_points*num_points_prompt points for other views')

    parser.add_argument("--min_depth_percentile",
                        help="minimum visualization depth percentile",
                        type=float, default=5)
    parser.add_argument("--max_depth_percentile",
                        help="maximum visualization depth percentile",
                        type=float, default=95)

    args = parser.parse_args()

    args.img_file_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

    return args


# noinspection PyPep8Naming
def main():
    args = parse()

    post_sam(args)


if __name__ == '__main__':
    main()
