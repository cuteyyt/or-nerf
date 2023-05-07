"""
Calculate metrics
"""

import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm


def cal_psnr(pred, target):
    psnr = compare_psnr(target, pred)
    return psnr


def test_render(pred_dir, target_dir):
    in_paths = [os.path.join(pred_dir, f) for f in sorted(os.listdir(pred_dir)) if f.endswith('png')]
    target_paths = [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir)) if f.endswith('png')]

    # Cal pixel-wise accuracy and iou for each img
    print(f'Test psnr for {target_dir}')
    psnr_all = list()
    with tqdm(total=len(in_paths)) as p_bar:
        for (in_path, target_path) in zip(in_paths, target_paths):
            pred = cv2.imread(in_path)
            target = cv2.imread(target_path)

            psnr = cal_psnr(pred, target)
            psnr_all.append(psnr)

            p_bar.update()

    with open(os.path.join(pred_dir, 'test_results.txt'), 'w') as f:
        for i, psnr in enumerate(psnr_all):
            f.write('img {}: psnr {:.3f}\n'.format(in_paths[i], psnr))

        print('scene: psnr {:.2f}'.format(sum(psnr_all) / len(psnr_all)))
        f.write('scene: psnr {:.3f}'.format(sum(psnr_all) / len(psnr_all)))
        f.close()


def cal_iou(pred, target, eps=1e-10):
    intersection = np.multiply(pred, target).reshape(-1, 1)
    union = np.asarray(pred + target > 0, np.float32).reshape(-1, 1)
    iou = intersection.sum() / (union.sum() + eps)

    return iou


def cal_pixel_wise_acc(pred, target):
    acc = len(np.transpose(np.nonzero((pred - target) == 0))) / (target.shape[0] * target.shape[1])
    return acc


def test_mask(pred_dir, target_dir):
    in_paths = [os.path.join(pred_dir, f) for f in sorted(os.listdir(pred_dir)) if f.endswith('png')]
    target_paths = [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir)) if f.endswith('png')]

    # Cal pixel-wise accuracy and iou for each img
    print(f'Test pixel-wise acc and iou for {target_dir}')
    acc_all = list()
    iou_all = list()
    with tqdm(total=len(in_paths)) as p_bar:
        for (in_path, target_path) in zip(in_paths, target_paths):
            pred = cv2.imread(in_path, -1)
            pred[pred == 255] = 1
            target = cv2.imread(target_path, -1)

            acc = cal_pixel_wise_acc(pred, target)
            acc_all.append(acc * 100.)

            iou = cal_iou(pred, target)
            iou_all.append(iou * 100.)

            p_bar.update()

    with open(os.path.join(pred_dir, 'test_results.txt'), 'w') as f:
        for i, (acc, iou) in enumerate(zip(acc_all, iou_all)):
            f.write('img {}: pa {:.3f}, iou {:.3f}\n'.format(in_paths[i], acc, iou))

        print('scene: pa {:.2f}, iou {:.2f}'.format(sum(acc_all) / len(acc_all), sum(iou_all) / len(iou_all)))
        f.write('scene: pa {:.3f}, iou {:.3f}'.format(sum(acc_all) / len(acc_all), sum(iou_all) / len(iou_all)))
        f.close()


def test(pred_dir, target_dir, mode):
    if mode == 'mask':
        test_mask(pred_dir, target_dir)
    elif mode == 'render':
        test_render(pred_dir, target_dir)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--mode', choices=['mask', 'render'], type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    pred_dir, target_dir = args.pred_dir, args.target_dir
    mode = args.mode

    test(pred_dir, target_dir, mode)


if __name__ == '__main__':
    main()
