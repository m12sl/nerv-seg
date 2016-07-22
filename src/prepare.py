from __future__ import print_function

import os
import numpy as np
import argparse
from tqdm import tqdm

import cv2

# TODO: move size into args
image_rows = 420
image_cols = 580


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
    path = os.path.join(args.data_path, 'processed/' + args.prefix + '{}')
    np.save(path.format('test'), imgs)
    np.save(path.format('test_idx'), idx)


def get_train_sequences(args):
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
        ret[a] = [t[1] for t in sorted(line, key=lambda t: t[0])]

    print('There are {} sequences of images'.format(len(ret)))
    return ret


def prepare_image_and_mask(path):
    mask_path = os.path.splitext(path)[0] + '_mask.tif'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.array(img), np.array(mask)


def stack_image_files(files):
    imgs = np.ndarray((len(files), 1, image_rows, image_cols), dtype=np.uint8)
    masks = np.ndarray((len(files), 1, image_rows, image_cols), dtype=np.uint8)
    for i, f in tqdm(enumerate(files)):
        img, mask = prepare_image_and_mask(f)
        imgs[i, 0, ...] = img
        masks[i, 0, ...] = mask
    return imgs, masks


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
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='path to data folder: raw and processed')

    parser.add_argument('--prefix', type=str, default='tmp-',
                        help='current run prefix')

    parser.add_argument('--num_folds', type=int, default=5,
                        help='number of folds')

    parser.add_argument('--seed', type=int, default=2019,
                        help='just data shuffling and fold splitting seed')

    args = parser.parse_args()
    prepare_test(args)
    build_folds(args)

if __name__ == '__main__':
    main()
