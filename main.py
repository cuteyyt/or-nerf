import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    # Convert sam label to spinnerf format
    in_dir = 'data\ibrnet_data_sam\qq11\lama'
    out_dir = 'data\statue\images_1\depth_ori\depth_label'

    in_paths = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if 'mask' in f]
    in_paths = sorted(in_paths)

    i = 0
    for f in tqdm(in_paths):
        in_path = f
        out_path = os.path.join(out_dir, 'img{:03d}.png'.format(i))

        # img = cv2.imread(in_path, -1)
        # img[img == 255] = 1
        #
        # cv2.imwrite(out_path, img)
        shutil.copy(in_path, out_path)
        i += 1

    mask_paths = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if 'mask' in f]
    mask_paths = sorted(mask_paths, key=lambda x: int(Path(x).name.split('_')[0][len('image'):]))

    in_paths = [os.path.join('', f) for f in os.listdir('data\ibrnet_data_sam\qq11\masks')]
    in_paths = sorted(in_paths)

    i = 0
    for mask_path in tqdm(mask_paths):
        out_path = os.path.join('data\statue\images_1\label', Path(in_paths[i]).name)

        mask = cv2.imread(mask_path, -1)
        mask[mask == 255] = 1
        cv2.imwrite(out_path, mask)
        # shutil.copy(mask_path, out_path)

        i += 1

    img = cv2.imread('data\statue\images_1\label\IMG_2707.png', -1)
    img = (img * 255).astype(np.uint8)
    cv2.imshow('debug', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
