"""
finds transformation that is required to find mismatch between tiff and jpg
"""
from __future__ import division
import os
import numpy as np
import cv2
from tqdm import tqdm


def stretch_n(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def allign_images(im1_gray, im2_gray, im2):  # Read the images to be aligned

    # Find size of image1
    sz = im1_gray.shape

    im1_gray = cv2.cvtColor(im1_gray[16:240, 16:240], cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_gray[16:240, 16:240], cv2.COLOR_BGR2GRAY)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 50000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), borderMode=cv2.BORDER_REPLICATE,
                                          flags=cv2.BORDER_REPLICATE + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
                im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                             borderMode=cv2.BORDER_REPLICATE,
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned, warp_matrix


if __name__ == '__main__':

    train_jpg_path = '../data/train-jpg'
    train_tiff_path = '../data/train-tif-v2'
    train_tiff_shifted_path = '../data/train-tif-shifted'

    try:
        os.mkdir(train_tiff_shifted_path)
    except:
        pass

    for file_name in tqdm(os.listdir(train_jpg_path)):
        img = cv2.imread(os.path.join(train_jpg_path, file_name))
        img_tif = cv2.imread(os.path.join(train_tiff_path, file_name.replace('.jpg', '.tif')))

        try:
            shifted_tiff, w = allign_images(stretch_n(img, 0, 100), stretch_n(img_tif[:, :, :3], 0, 100), img_tif)
        except:
            print file_name
            shifted_tiff = img_tif

        cv2.imwrite(os.path.join(train_tiff_shifted_path, file_name), shifted_tiff)
