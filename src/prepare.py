from __future__ import print_function

import os
import numpy as np
import argparse
from tqdm import tqdm
from time import time

import cv2

# TODO: move size into args
image_rows = 420
image_cols = 580


def proc(X, args):
    Y = np.ndarray((X.shape[0], X.shape[1],
                    args.height, args.width))

    for i in range(X.shape[0]):
        Y[i, 0] = cv2.resize(X[i, 0], (args.width, args.height),
                             interpolation=cv2.INTER_CUBIC)
    return Y


def prepare_dataset(args):
    t0 = time()
    print('Load train images')
    img, mask, idx = prepare_train(args)
    # img is a stack of training images, (total_images, 1, W, H)
    # mask is the same stack of masks
    # idx -- index for image_id -> (subject, number)
    print('Process and save: bulk train')
    path = os.path.join(args.data_path, 'common/{}_{}.npy')

    np.save(path.format('train', 'img'), img)
    np.save(path.format('train', 'mask'), mask)
    np.save(path.format('train', 'idx'), idx)

    img = proc(img, args)
    mask = proc(mask, args).astype(np.float32) / 255.0

    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()

    img -= mean
    img /= std

    size = '{}x{}'.format(args.width, args.height)
    path = os.path.join(args.data_path, 'processed/{}_{}_{}.npy')
    np.save(path.format('train', 'img', size), img)
    np.save(path.format('train', 'mask', size), mask)
    # TODO: add index
    print('Load test images')
    test, test_idx = prepare_test(args)

    print('Process and save: test images')
    path = os.path.join(args.data_path, 'common/{}_{}.npy')
    np.save(path.format('test', 'img'), test)
    np.save(path.format('test', 'idx'), test_idx)

    test = proc(test, args)
    test = test.astype(np.float32)
    test -= mean
    test /= std

    path = os.path.join(args.data_path, 'processed/{}_{}_{}.npy')
    np.save(path.format('test', 'img', size), test)

    t1 = time()
    print('Done for {:.2f}m'.format((t1 - t0) / 60.0))


def get_test_sequence(args):
    root = os.path.join(args.data_path, 'raw/test')
    files = os.listdir(root)

    return [os.path.join(root, f) for f in files]


def prepare_test(args):
    import re
    files = get_test_sequence(args)
    imgs = np.ndarray((len(files), 1, image_rows, image_cols), dtype=np.uint8)

    print('Prepare test set')
    idx = np.zeros((len(files),), dtype=np.int32)
    for i, f in tqdm(enumerate(files)):
        r = re.search('(\d+).tif', f)
        if r:
            idx[i] = int(r.group(1))
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            imgs[i, 0, ...] = img
    return imgs, idx


def prepare_train(args):
    import re
    root = os.path.join(args.data_path, 'raw/train')
    files = os.listdir(root)

    gpby = {}
    for fname in files:
        r = re.match('(\d+)_(\d+).tif', fname)
        if r:
            a, b = [int(t) for t in r.groups()]
            if a not in gpby:
                gpby[a] = []
            gpby[a].append((b, os.path.join(root, fname)))

    ret = {}
    for a, line in gpby.items():
        ret[a] = sorted(line, key=lambda t: t[0])

    total = sum([len(t) for t in ret.values()])

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    masks = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    idx = np.zeros((total, 2))
    
    counter = 0
    
    for subj, line in ret.items():
        for i, fname in line:
            img, mask = prepare_image_and_mask(fname)
            imgs[counter, 0, ...] = img
            masks[counter, 0, ...] = mask
            idx[counter, :] = [subj, i]
            counter += 1

    return imgs, masks, idx


def prepare_image_and_mask(path):
    mask_path = os.path.splitext(path)[0] + '_mask.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.array(img), np.array(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='path to data folder: raw and processed')

    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--height', type=int, default=64)

    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == '__main__':
    main()
