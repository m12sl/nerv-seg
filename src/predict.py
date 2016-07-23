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


def load_test_data(args):
    img = preprocess(np.load(args.test), args).astype(np.float32)
    img -= img.mean()
    img /= np.std(img)
    return img


def predict(args):
    print('Load test data')
    img = load_test_data(args)
    print('Creating and compiling model...')
    model = Model(args)
    model.load_weights(args.model)
    print('Predicting masks on test data...')
    mask = model.predict(img, verbose=1)
    np.save(args.save_to, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_width', type=int, default=80,
                        help='image width')
    parser.add_argument('--img_height', type=int, default=64,
                        help='image height')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')

    parser.add_argument('--model', type=str, 
                        default='../models/tmp/ckpt_best.hdf5',
                        help='path to model weights')
    parser.add_argument('--test', type=str,
                        default='../data/processed/tmp-test.npy',
                        help='path to bulk of test imgs')
    parser.add_argument('--save_to', type=str,
                        default='../data/results/tmp-res.npy')
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
