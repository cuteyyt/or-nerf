import cv2
import numpy as np
import pyclipper


def scale_by_pyclipper(points, scale_size=1.):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)  # JT_ROUND
    scaled_poly = pco.Execute(int(scale_size))
    pco.Clear()

    scaled_poly = np.asarray(scaled_poly)
    return scaled_poly


def mask_refine(mask, **kwargs):
    # For finding contours, we need channel c set to 1
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    # Step 1. Filter irregular masks
    if 'num_masks' in kwargs:
        mask_filtered = np.zeros_like(mask_gray)
        contours, hierarchies = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = list()
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            contour_areas.append(contour_area)
        contour_areas = np.asarray(contour_areas)

        contours_refined = list()
        contours_indices = np.argsort(contour_areas)[-1::-1][:kwargs['num_masks']]
        for idx in contours_indices:
            contours_refined.append(contours[idx])

        cv2.drawContours(mask_filtered, contours_refined, -1, (255, 255, 255), cv2.FILLED)
    else:
        mask_filtered = np.copy(mask_gray)

    # Step 2. Dilate to smooth the borders, fill-in holes
    mask_dilated = np.copy(mask_filtered)
    if 'dilate_kernel_size' and 'dilate_iters' in kwargs:
        dilate_kernel_size = kwargs['dilate_kernel_size']
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        mask_dilated = cv2.dilate(mask_dilated, dilate_kernel, iterations=kwargs['dilate_iters'])

    # Step 3. Scale the contour
    if 'contour_scale_size' in kwargs:
        mask_scaled = np.zeros_like(mask_dilated)
        contours, hierarchies = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) == kwargs['num_masks']

        for contour in contours:
            contour_scaled = scale_by_pyclipper(contour.squeeze(1), scale_size=kwargs['contour_scale_size'])
            cv2.drawContours(mask_scaled, contour_scaled, -1, (255, 255, 255), -1)
    else:
        mask_scaled = np.copy(mask_dilated)

    mask_refined = mask_scaled.copy()
    return mask_refined
