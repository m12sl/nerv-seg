import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def elastic_transform(X, y=None, alpha=None, sigma=None, alpha_affine=None, random_state=None):
    alpha = X.shape[1] * 2 if alpha is None else alpha
    sigma = X.shape[1] * 0.08 if sigma is None else sigma
    alpha_affine = X.shape[1] * 0.08 if alpha_affine is None else alpha_affine
    random_state = np.random.RandomState(None) if random_state is None else random_state

    shape_size = X.shape
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, \
                       [center_square[0] + square_size, center_square[1] - square_size], \
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape)\
                                      .astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    X = cv2.warpAffine(X, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    if y is not None:
        y = cv2.warpAffine(y, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # Elastic
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), \
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), \
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    xx, yy = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    xx = np.reshape(xx + dx, (-1, 1)).astype('float32')
    yy = np.reshape(yy + dy, (-1, 1)).astype('float32')

    if y is not None:
        return (cv2.remap(X, xx, yy, \
                          interpolation=cv2.INTER_LINEAR, \
                          borderMode=cv2.BORDER_REFLECT).reshape(shape_size),
                cv2.remap(y, xx, yy, \
                          interpolation=cv2.INTER_LINEAR, \
                          borderMode=cv2.BORDER_REFLECT).reshape(shape_size))
    return cv2.remap(X, xx, yy, interpolation=cv2.INTER_LINEAR, \
                    borderMode=cv2.BORDER_REFLECT).reshape(shape_size)


# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def test_elastic():
    import time

    # Load images
    im = cv2.imread("../data/raw/train/10_1.tif", -1)
    im_mask = cv2.imread("../data/raw/train/10_1_mask.tif", -1)

    # Draw grid lines
    draw_grid(im, 50)
    draw_grid(im_mask, 50)

    start = time.time()
    # Apply transformation on image
    im_t, im_mask_t = elastic_transform(X=im, \
                                        y=im_mask, \
                                        alpha=im.shape[1] * 2, \
                                        sigma=im.shape[1] * 0.08, \
                                        alpha_affine=im.shape[1] * 0.08)
    print("Transformed in %.2f sec" % (time.time() - start))

    # Display result
    plt.figure(figsize = (16,14))
    plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
    plt.savefig("transformed.png")


if __name__=="__main__":
    test_elastic()

