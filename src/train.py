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


def preprocess(imgs, args):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1],
                         args.img_height, args.img_width),
                        dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (args.img_width, args.img_height), 
                                  interpolation=cv2.INTER_CUBIC)
    return imgs_p


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

    img_train, mask_train = load_data('train', args)
    img_val, mask_val = load_data('val', args)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = Model(args)
    ckpt_best = ModelCheckpoint(os.path.join(args.save_dir, 'ckpt_best.hdf5'),
                                monitor='val_loss', save_best_only=True)
    ckpt = ModelCheckpoint(os.path.join(args.save_dir, 'ckpt.hdf5'),
                           monitor='val_loss', save_best_only=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(img_train, mask_train,
              validation_data=(img_val, mask_val),
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              verbose=1, shuffle=True,
              callbacks=[ckpt, ckpt_best])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('Not implemented')
    print('-'*30)

    # imgs_test, imgs_id_test = load_test_data(args)

    # imgs_test = preprocess(imgs_test, args)

    # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # model.load_weights(os.path.join(args.save_dir, 'best_unet.hdf5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # imgs_mask_test = model.predict(imgs_test, verbose=1)

    # TODO: rewrite this
    #  np.save(os.path.join(args.data_path, 'processed/imgs_mask_test.npy'), imgs_mask_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='path to data folder: raw and processed')
    parser.add_argument('--save_dir', type=str, default='../models/tmp',
                        help='directory to store checkpointed models')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                'config.pkl'        : configuration;
                'chars_vocab.pkl'   : vocabulary definitions;
                'checkpoint'        : paths to model file(s) (created by tf).
                                      Note: this file contains absolute paths, be careful when moving files around;
                'model.ckpt-*'      : file(s) with model definition (created by tf)
                """)

    parser.add_argument('--img_width', type=int, default=80,
                        help='image width')
    parser.add_argument('--img_height', type=int, default=64,
                        help='image height')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')

    # yes, masks path will be compute in dataloader
    parser.add_argument('--train', type=str, 
                        default='../data/processed/tmp-train_0_5_img.npy',
                        help='path to bulk of train imgs npy')
    parser.add_argument('--val', type=str,
                        default='../data/processed/tmp-val_0_5_img.npy',
                        help='path to bulk of val imgs npy')
    parser.add_argument('--test', type=str,
                        default='../data/processed/tmp-test.npy',
                        help='path to bulk of test imgs')
    args = parser.parse_args()
    train_and_predict(args)


if __name__ == '__main__':
    main()
