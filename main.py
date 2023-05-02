import cv2
import numpy as np


def main():
    img = cv2.imread('data\ibrnet_data_spinnerf\qq3\images_1\label\img000.png', -1)
    img = (img * 255).astype(np.uint8)
    cv2.imshow('debug', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
