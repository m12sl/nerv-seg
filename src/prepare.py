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
#    print('List train set')
#    train_seq = get_train_sequences(args)
#    names = np.concatenate(list(train_seq.values()))
    print('Load train images')
#    img, mask = stack_image_files(names)
    img, mask, idx = prepare_train(args)

    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()

    img -= mean
    img /= std
    print('Process and save: bulk train')
    path = os.path.join(args.data_path, 'processed/{}_{}_{}.npy')

    if args.keep_full:
        np.save(path.format('train', 'img', 'full'), img)
        np.save(path.format('train', 'mask', 'full'), mask)

    img = proc(img, args)
    mask = proc(mask, args).astype(np.float32) / 255.0
    
    size = '{}x{}'.format(args.width, args.height)
    np.save(path.format('train', 'idx', ''), idx)
    np.save(path.format('train', 'img', size), img)
    np.save(path.format('train', 'mask', size), mask)

    # TODO: add index
    print('Load test images')
    test, test_idx = prepare_test(args)
    test = test.astype(np.float32)

    test -= mean
    test /= std
    print('Process and save: test images')
    if args.keep_full:
        np.save(path.format('test', 'img', 'full'), test)

    test = proc(test, args)
    np.save(path.format('test', 'img', size), test)
    np.save(path.format('test', 'idx', ''), test_idx)
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


def build_folds(args):
    # TODO: (TBD) may be reserve one fold for ensembling?
    # TODO: if we will try some folding tricks, build only
    #       one bulk_img/bulk_mask array and move
    # indexing procedures into fold_i_j_idx.npy
    # for usage like `fold_i = bulk[fold_i_idx, ...] `
    train_sequences_dict = get_train_sequences(args)

    keys = list(train_sequences_dict)
    idx = np.array(keys)
    np.random.seed(args.seed)
    np.random.shuffle(idx)

    idx_folds = np.array_split(idx, args.num_folds)
    fname = os.path.join(args.data_path, 'processed/',
                         args.prefix + '{}_{}_{}_{}.npy')
    print('Generate {} folds'.format(args.num_folds))

    for i in range(args.num_folds):
        val_idx = idx_folds[i]
        train_idx = np.concatenate(idx_folds[:i] + idx_folds[i + 1:])

        train_files, val_files = [], []
        [train_files.extend(train_sequences_dict[j]) for j in train_idx]
        [val_files.extend(train_sequences_dict[j]) for j in val_idx]
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)

        print("{}: {} {}".format(i, len(train_files), len(val_files)))
        imgs, masks = stack_image_files(train_files)
        np.save(fname.format('train', i, args.num_folds, 'img'), imgs)
        np.save(fname.format('train', i, args.num_folds, 'mask'), imgs)

        imgs, masks = stack_image_files(val_files)
        np.save(fname.format('val', i, args.num_folds, 'img'), imgs)
        np.save(fname.format('val', i, args.num_folds, 'mask'), imgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='path to data folder: raw and processed')

    parser.add_argument('--keep-full', default=False, action='store_true',
                        help='save full size images')

    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--height', type=int, default=64)

    args = parser.parse_args()
    prepare_dataset(args)
    # TODO: repair fold preparation
#    prepare_test(args)
#    build_folds(args)

if __name__ == '__main__':
    main()
