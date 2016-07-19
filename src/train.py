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
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], args.img_height, args.img_width), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (args.img_width, args.img_height), interpolation=cv2.INTER_CUBIC)
    return imgs_p


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
    imgs_train, imgs_mask_train = load_train_data(args)

    imgs_train = preprocess(imgs_train, args)
    imgs_mask_train = preprocess(imgs_mask_train, args)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = Model(args)
    model_checkpoint = ModelCheckpoint(os.path.join(args.save_dir, 'best_unet.hdf5'), monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=args.batch_size, nb_epoch=args.num_epochs, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data(args)
    imgs_test = preprocess(imgs_test, args)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(os.path.join(args.save_dir, 'best_unet.hdf5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    # TODO: rewrite this
    np.save(os.path.join(args.data_path, 'processed/imgs_mask_test.npy'), imgs_mask_test)


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

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')

    args = parser.parse_args()
    train_and_predict(args)


if __name__ == '__main__':
    main()