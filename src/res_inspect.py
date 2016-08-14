import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_doh
import seaborn as sns
PREDICTIONS = '../data/predict-0/'


def load_fold(no):
    test = np.load(PREDICTIONS + 'fold-{}test.npy'.format(no))
    train = np.load(PREDICTIONS + 'fold-{}train.npy'.format(no))
    return train, test


def load_gnd():
    mask = np.load('../data/processed/train_mask_80x64.npy')
    return mask


def dice(x, y):
    smooth = 1.0
    intersection = np.sum(x.flatten() * y.flatten())
    denom = np.sum(x.flatten() + y.flatten())
    return (2.0 * intersection + smooth) / (denom + smooth)


def plot(A, B, i):
    blobs_doh = blob_doh(A[i, 0, ...], max_sigma=30, threshold=.01)
    plt.figure()
    ax = plt.subplot(211)
    cax = ax.imshow(A[i, 0, ...], interpolation="nearest", cmap="cubehelix")
    for blob in blobs_doh:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.colorbar(cax)

    ax = plt.subplot(212)
    ax.imshow(B[i, 0, ...], interpolation="nearest", cmap="cubehelix")

    plt.show()