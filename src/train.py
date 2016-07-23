from __future__ import print_function

import cv2
import numpy as np

import argparse
import os
import errno
import pickle
import json
from tqdm import tqdm

from model import UNet as Model
from data import load_train_data, load_test_data

from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def load_data(kind, args):
    if kind == 'train':
        path = args.train
    elif kind == 'val':
        path = args.val
    else:
        raise ValueError(kind)

    img = preprocess(np.load(path), args).astype(np.float32)
    # just let it be ")
    mask = preprocess(np.load(path[:-7] + 'mask.npy'), args).astype(np.float32)

    img -= img.mean()
    img /= np.std(img)
    mask /= 255.0
    return img, mask


def train_and_predict(args):
    path = args.save_dir
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

#    img_train, mask_train = load_data('train', args)
#    img_val, mask_val = load_data('val', args)
    img_train = np.load(args.bulk)
    mask_train = np.load(args.bulk.replace('_img_', '_mask_'))

    img_train = np.load('../data/processed/fold_0/train_img_.npy')
    mask_train = np.load('../data/processed/fold_0/train_mask_.npy')
    img_val = np.load('../data/processed/fold_0/val_img_.npy')
    mask_val = np.load('../data/processed/fold_0/val_mask_.npy')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = Model(args)
    best_path = os.path.join(args.save_dir, 'ckpt_best.hdf5')
    ckpt_best = ModelCheckpoint(best_path,
                                monitor='val_loss', save_best_only=True)
#    ckpt = ModelCheckpoint(os.path.join(args.save_dir, 'ckpt.hdf5'),
#                           monitor='val_loss', save_best_only=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(img_train, mask_train,
              validation_data=(img_val, mask_val),
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              verbose=1, shuffle=True,
              callbacks=[ckpt_best])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    test = np.load(args.test)
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(best_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    mask = model.predict(test, verbose=1)
    np.save(os.path.join(args.data_path, 'results/pred.npy'), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='path to data folder: raw and processed')
    parser.add_argument('--save_dir', type=str, default='../models/tmp',
                        help='directory to store checkpointed models')
    parser.add_argument('--img_width', type=int, default=80,
                        help='image width')
    parser.add_argument('--img_height', type=int, default=64,
                        help='image height')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')

    # yes, masks path will be compute in dataloader
    parser.add_argument('--bulk', type=str, 
                        default='../data/processed/train_img_80x64.npy',
                        help='path to bulk of train imgs npy')
    parser.add_argument('--val', type=str,
                        default='../data/processed/val_0_5_img.npy',
                        help='path to bulk of val imgs npy')
    parser.add_argument('--test', type=str,
                        default='../data/processed/test_img_80x64.npy',
                        help='path to bulk of test imgs')
    args = parser.parse_args()
    train_and_predict(args)


if __name__ == '__main__':
    main()
