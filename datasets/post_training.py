import argparse
import os

import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def cal_psnr(pred_dir, target_dir):
    pred_paths = [os.path.join(pred_dir, f) for f in pred_dir]
    pred_paths = sorted(pred_paths)

    target_paths = [os.path.join(target_dir, f) for f in target_dir]
    target_paths = sorted(target_paths)

    psnr_all = 0.
    for pred_path, target_path in zip(pred_paths, target_paths):
        pred = cv2.imread(pred_path)
        target = cv2.imread(target_path)
        psnr = compare_psnr(target, pred)
        psnr_all += psnr
    psnr_mean = psnr_all / len(pred_paths)

    print(f'psnr mean of {pred_dir} {len(pred_paths)} imgs is: {psnr_mean}')
    return psnr_mean


def rgb2bgr(in_dir, out_dir):
    for f in os.listdir(in_dir):
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f)

        img = cv2.imread(in_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, img)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    in_dir, out_dir = args.in_dir, args.out_dir

    # rgb2bgr(in_dir, out_dir)
    cal_psnr(in_dir, out_dir)


if __name__ == '__main__':
    main()
