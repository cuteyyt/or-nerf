import configargparse
from tqdm import tqdm
import numpy as np
import cv2
from glob import glob
import os
import copy
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

LABEL = 'label_mv_bootstrapped'
REFINED_PATH = 'refined_images_mv_bootstrapped'
REFINED_DEPTH_PATH = 'refined_disp_mv_bootstrapped'


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--dataset", type=str, default=None, help='dataset name')
    parser.add_argument("--dilate_iters", type=int, default=5, help='dilation iterations')
    parser.add_argument("--alpha_thresh", type=float, default=0.1,
                        help='threshold on alphas to filter high density points')
    parser.add_argument("--N_gt", type=int, default=40, help='number of ground truth images in the dataset')
    parser.add_argument("--distance_thresh", type=float, default=0.01, help='a threshold for distance similarity')
    return parser


def main(args):
    MASK_DIR = f'data/test/Scenes/{args.dataset}/images_4/{LABEL}'
    DATA_DIR = f'logs/{args.dataset}/renderonly_train_003999/'
    DILATE_ITERS = args.dilate_iters

    ALPHA_THRESH = args.alpha_thresh
    N_gt = args.N_gt
    DISTANCE_THRESH = args.distance_thresh

    def convert_pose(C2W):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        C2W = np.matmul(C2W, flip_yz)
        return C2W

    n_imgs = len(glob(f"./{DATA_DIR}/rgb/*.png"))
    images = [Image.open(f'./{DATA_DIR}/images/{idx:06}.png') for idx in range(n_imgs)]
    print(np.array(images[0]).shape)
    masks = [np.array(Image.open(x)) for x in sorted(glob(os.path.join(MASK_DIR, '*.png')))[-n_imgs:]]
    masks = [(x / x.max() > 0.5).astype('uint8') for x in masks]
    masks = [cv2.dilate(np.array(msk), np.ones((5, 5), np.uint8), iterations=DILATE_ITERS) for msk in masks]
    for i in range(len(masks)):
        if masks[i].shape[-1] == 3:
            masks[i] = masks[i][:, :, 0]
    print(masks[0].shape)
    zs = [np.load(f'./{DATA_DIR}/z/{idx:06}.npy') for idx in range(n_imgs)]
    weights = [np.load(f'./{DATA_DIR}/weight/{idx:06}.npy') for idx in range(n_imgs)]
    alphas = [np.load(f'./{DATA_DIR}/alpha/{idx:06}.npy') for idx in range(n_imgs)]
    alpha_threshs = [x.max(axis=-1) * ALPHA_THRESH for x in alphas]
    c2ws = [np.loadtxt(f'./{DATA_DIR}/pose/{idx:06}.txt') for idx in range(n_imgs)]
    K = np.loadtxt(f'./{DATA_DIR}/intrinsics.txt')
    K_inv = np.linalg.inv(K)
    depths = [np.load(f'./{DATA_DIR}/depth/{idx:06}.npy') for idx in range(n_imgs)]
    disps = [np.load(f'./{DATA_DIR}/disp/{idx:06}.npy') for idx in range(n_imgs)]
    H, W = masks[0].shape[0:2]
    print(H, W)

    project2world_cache = {}

    def project2world(uv_A, z_A, c2w_A, c2w_A_inv, K, K_inv):
        key = str(src_idx) + str(uv_A)
        pt_z_A = z_A[min(uv_A[1], z_A.shape[0] - 1), min(uv_A[0], z_A.shape[1] - 1)][
                     np.logical_and(src_alpha[min(uv_A[1], z_A.shape[0] - 1),
                                              min(uv_A[0], z_A.shape[1] - 1)] >= alpha_threshs[src_idx][
                                        min(uv_A[1], z_A.shape[0] - 1), min(uv_A[0], z_A.shape[1] - 1)],
                                    src_z[min(uv_A[1], z_A.shape[0] - 1), min(uv_A[0], z_A.shape[1] - 1)] >=
                                    depths[src_idx][min(uv_A[1], z_A.shape[0] - 1), min(uv_A[0], z_A.shape[1] - 1)]
                                    )
                 ][None, :, None]  # (1, n_depths, 1)

        if key in project2world_cache:
            return project2world_cache[key], pt_z_A.reshape(-1)

        n_depths = pt_z_A.shape[1]
        xyz_A_camera = (np.stack([uv_A[0], uv_A[1], 1])[None, None, :] * pt_z_A) @ np.linalg.inv(K).T
        # xyz_A_camera has shape (1, n_depths, 3).

        xyz_A_world = np.concatenate([xyz_A_camera, np.ones([1, n_depths, 1])], axis=2) @ c2w_A.T
        # xyz_A_world has shape (1, n_depths, 4).

        project2world_cache[key] = xyz_A_world
        return xyz_A_world, pt_z_A.reshape(-1)

    def reprojection(uv_A, z_A, c2w_A, c2w_A_inv, c2w_B, c2w_B_inv, K, K_inv):
        """
        Args
            uvs_A: of shape (n_uvs, 2)
        """
        xyz_A_world, pt_z_A = project2world(uv_A, z_A, c2w_A, c2w_A_inv, K, K_inv)
        # xyz_A_world has shape (1, n_depths, 4).

        uvs_B = (xyz_A_world @ c2w_B_inv.T)[:, :, :3] @ K.T
        zs_B = uvs_B[:, :, 2:].reshape(-1)
        uvs_B = (uvs_B[:, :, :2] / uvs_B[:, :, 2:]).astype(np.int32)
        return uvs_B[0], zs_B, pt_z_A

    def project_depth(uv_A, z_A, c2w_A, c2w_A_inv, c2w_B, c2w_B_inv, K, K_inv):
        pt_z_A = np.array([z_A])[None, ..., None]

        n_depths = pt_z_A.shape[1]
        xyz_A_camera = (np.stack([uv_A[0], uv_A[1], 1])[None, None, :] * pt_z_A) @ np.linalg.inv(K).T
        # xyz_A_camera has shape (1, n_depths, 3).

        xyz_A_world = np.concatenate([xyz_A_camera, np.ones([1, n_depths, 1])], axis=2) @ c2w_A.T
        # xyz_A_world has shape (1, n_depths, 4).

        uvs_B = (xyz_A_world @ c2w_B_inv.T)[:, :, :3] @ K.T
        zs_B = uvs_B[:, :, 2:]
        return zs_B[0, 0, 0]

    def unmasked_counterparts(u_A, v_A, draw=False):
        uvs_B, zs_B, pt_z_A = reprojection([u_A, v_A], src_z, src_c2w, src_c2w_inv, tgt_c2w, tgt_c2w_inv, K, K_inv)
        for i in range(len(uvs_B)):
            u_B, v_B = tuple(uvs_B[i])
            z_B = zs_B[i]

            try:
                if tgt_msk[v_B, u_B] == 0:
                    #                     idx_B = np.argmin(np.abs(tgt_z[v_B, u_B] - z_B))
                    #                     if np.max(tgt_weight[v_B, u_B][max(0, idx_B - 1): idx_B + 2]) == max_weights[tgt_idx][v_B, u_B]:
                    #                         return (u_B, v_B), pt_z_A[i]
                    if abs(z_B - 1 / tgt_disp[v_B, u_B]) / z_B < DISTANCE_THRESH:
                        projected_z = project_depth([u_B, v_B], 1 / tgt_disp[v_B, u_B], tgt_c2w, tgt_c2w_inv, src_c2w,
                                                    src_c2w_inv, K, K_inv)
                        return (u_B, v_B), projected_z
            except:
                pass
        return None, None

    REFINED_DIR = os.path.join(MASK_DIR, f'../{REFINED_PATH}')
    REFINED_DISP_DIR = os.path.join(MASK_DIR, f'../{REFINED_DEPTH_PATH}')
    REFINED_MASK_DIR = os.path.join(MASK_DIR, f'../{REFINED_PATH}/label')
    os.makedirs(REFINED_DIR, exist_ok=True)
    os.makedirs(REFINED_DISP_DIR, exist_ok=True)
    os.makedirs(REFINED_MASK_DIR, exist_ok=True)

    file_names = sorted(glob(os.path.join(MASK_DIR, '../../images/*.*')))[N_gt:]
    file_names = [x.split('/')[-1].replace('jpg', 'png') for x in file_names]

    refined_images = []
    refined_masks = []
    refined_disps = []
    for src_idx in range(0, n_imgs):
        print("_____________________")
        print("Source index:", src_idx)

        src_msk = copy.deepcopy(masks[src_idx])
        src_img = copy.deepcopy(images[src_idx])
        src_z = zs[src_idx]
        src_alpha = alphas[src_idx]
        src_weight = weights[src_idx]
        src_c2w = c2ws[src_idx]
        src_c2w = convert_pose(src_c2w)
        src_c2w_inv = np.linalg.inv(src_c2w)
        src_img_tensor = transforms.ToTensor()(src_img)
        numpy_src_img = np.array(src_img)
        src_disp = copy.deepcopy(disps[src_idx])

        rng = range(n_imgs - 1, -1, -1) if src_idx == 0 else [0]
        # rng = range(n_imgs - 1, -1, -1)
        for tgt_idx in tqdm(rng):
            if tgt_idx == src_idx:
                continue

            tgt_msk = copy.deepcopy(masks[tgt_idx])
            tgt_img = copy.deepcopy(images[tgt_idx])
            tgt_z = zs[tgt_idx]
            tgt_alpha = alphas[tgt_idx]
            tgt_weight = weights[tgt_idx]
            tgt_c2w = c2ws[tgt_idx]
            tgt_disp = copy.deepcopy(disps[tgt_idx])

            masked_coords = np.where(np.array(masks[src_idx]) == 1)

            # Convert poses from OpenCV to OpenGL
            tgt_c2w = convert_pose(tgt_c2w)
            tgt_c2w_inv = np.linalg.inv(tgt_c2w)
            W, H = src_img.size
            tgt_img_tensor = transforms.ToTensor()(tgt_img)
            numpy_tgt_img = np.array(tgt_img)

            for v_A, u_A in zip(list(masked_coords[0]), list(masked_coords[1])):
                out, z_val = unmasked_counterparts(u_A, v_A)
                if out is not None:
                    if src_msk[v_A, u_A] == 1 or src_disp[v_A, u_A] < 1 / z_val:
                        neighbor_dist = min(
                            abs(1 / z_val - src_disp[max(0, min(H - 1, v_A) - 1), min(W - 1, u_A)]),
                            abs(1 / z_val - src_disp[min(H - 1, v_A + 1), min(W - 1, u_A)]),
                            abs(1 / z_val - src_disp[min(H - 1, v_A), max(0, u_A - 1)]),
                            abs(1 / z_val - src_disp[min(H - 1, v_A), min(W - 1, u_A + 1)]),
                            abs(1 / z_val - src_disp[max(0, min(H - 1, v_A) - 1), max(0, u_A - 1)]),
                            abs(1 / z_val - src_disp[max(0, min(H - 1, v_A) - 1), min(W - 1, u_A + 1)]),
                            abs(1 / z_val - src_disp[min(H - 1, v_A + 1), max(0, u_A - 1)]),
                            abs(1 / z_val - src_disp[min(H - 1, v_A + 1), min(W - 1, u_A + 1)])
                        )
                        if neighbor_dist < DISTANCE_THRESH:
                            numpy_src_img[min(H - 1, v_A), min(W - 1, u_A)] = numpy_tgt_img[out[1], out[0]]
                            src_msk[min(H - 1, v_A), min(W - 1, u_A)] = 0
                            src_disp[min(H - 1, v_A), min(W - 1, u_A)] = 1 / z_val

        refined_images.append(Image.fromarray(numpy_src_img.astype('uint8'), 'RGB'))
        refined_masks.append(src_msk)
        refined_disps.append(src_disp)

        # masks[src_idx] = refined_masks[-1]
        # images[src_idx] = refined_images[-1]
        # disps[src_idx] = refined_disps[-1]

        refined_images[src_idx].save(os.path.join(REFINED_DIR, file_names[src_idx]))

        tmp = Image.fromarray((refined_masks[src_idx] * 255)[..., None].repeat(3, axis=-1).astype('uint8'), 'RGB')
        tmp.save(os.path.join(REFINED_MASK_DIR, file_names[src_idx]))

        tmp = Image.fromarray((refined_disps[src_idx] * 255)[..., None].repeat(3, axis=-1).astype('uint8'), 'RGB')
        tmp.save(os.path.join(REFINED_DISP_DIR, file_names[src_idx]))


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
