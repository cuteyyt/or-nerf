import os
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def rgb2bgr(in_dir, out_dir):
    for f in os.listdir(in_dir):
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f)

        img = cv2.imread(in_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, img)


def imgs2video(in_dir, out_path, img_file_type, fps=10):
    print(f'Cat images from {in_dir} to video {out_path}')
    # path = glob.glob(os.path.join(in_dir, '*.png'))
    img_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.endswith(img_file_type)]
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

    img_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.endswith(img_file_type)]
    img_paths = sorted(img_paths)

    n_imgs = len(img_paths)

    # Make the grid to ~16:9
    target_width = int(np.sqrt(n_imgs) * 4 / 3)
    # target_height = n_imgs // target_width

    img_list = list()
    for file in tqdm(img_paths):
        in_path = os.path.join(in_dir, file)

        img = cv2.imread(in_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    img_list = np.stack(img_list)  # B, H, W, C

    img_list = torch.from_numpy(img_list).float()
    img_list = torch.permute(img_list, (0, 3, 1, 2))
    img_grid = make_grid(img_list, nrow=target_width, normalize=True, padding=0)
    save_image(img_grid, out_path)


def down_sample_imgs(src_dir, target_dir, down_factor, img_suffix):
    print('Minifying', down_factor, (Path(src_dir)).parent.name)

    cwd = os.getcwd()
    resize_arg = '{}%'.format(100. / down_factor)

    check_output('cp {}/* {}'.format(src_dir, target_dir), shell=True)

    args = ' '.join(['mogrify', '-resize', resize_arg, '-format', 'png', '*.{}'.format(img_suffix)])
    print(args)
    os.chdir(target_dir)
    check_output(args, shell=True)
    os.chdir(cwd)

    if img_suffix != 'png':
        check_output('rm {}/*.{}'.format(target_dir, img_suffix), shell=True)
        print('Removed duplicates')
    print('Done')


# noinspection PyProtectedMember
def exif_transpose(img):
    """
    If an image has an Exif Orientation tag, transpose the image
    accordingly.

    Note: Very recent versions of Pillow have an internal version
    of this function. So this is only needed if Pillow isn't at the
    latest version.

    :param img: The image to transpose.
    :return: An image.
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image: nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


# noinspection PyTypeChecker
def load_img_exif(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    Defaults to returning the image data as a 3-channel array of 8-bit data. That is
    controlled by the mode parameter.

    Supported modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

    :param file: image file name or file object to load
    :param mode: format to convert the image to 'RGB' (8-bit RGB, 3 channels), 'L' (black and white)
    :return: image contents as numpy array
    """

    # Load the image with PIL
    img = Image.open(file)

    if hasattr(ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)


def abandon_exif(img, out_path):
    data = list(img.getdata())
    img_without_exif = Image.new(img.mode, img.size)
    img_without_exif.putdata(data)

    img_without_exif.save(out_path)

    return img_without_exif
