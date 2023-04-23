import os

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def imgs2video(in_dir, out_path, img_file_type, fps=10):
    print(f'Cat images from {in_dir} to video {out_path}')
    # path = glob.glob(os.path.join(in_dir, '*.png'))
    img_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.lower().endswith(img_file_type)]
    img_paths = sorted(img_paths)

    first_img = cv2.imread(img_paths[0])
    size = first_img.shape[:2][::-1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, size)

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        video.write(img)

    video.release()


def imgs2grid(in_dir, out_path, img_file_type=None):
    print(f'Merge images from {in_dir} to grid {out_path}')

    img_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.lower().endswith(img_file_type)]
    n_imgs = len(img_paths)

    # Make the grid to ~16:9
    target_width = int(np.sqrt(n_imgs) * 4 / 3)
    # target_height = n_imgs // target_width

    img_list = list()
    for file in tqdm(img_paths):
        in_path = os.path.join(in_dir, file)

        img = cv2.imread(in_path)
        # h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    img_list = np.stack(img_list)  # B, H, W, C
    # print(img_list.shape)

    img_list = torch.from_numpy(img_list).float()
    img_list = torch.permute(img_list, (0, 3, 1, 2))
    img_grid = make_grid(img_list, nrow=target_width, normalize=True, padding=0)
    save_image(img_grid, out_path)
