from __future__ import print_function

import numpy as np

import argparse
import os
import errno
import json

from generator import MyGenerator

from model import UNet
from keras.callbacks import ModelCheckpoint, EarlyStopping


def maybe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train(args):
    maybe_mkdir(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fout:
        json.dump(args.__dict__, fout)

    print('-'*30)
    print('Loading and preprocessing train data...')
    train = np.load(args.train)
    val = np.load(args.val)

    mean = np.mean(train[:, [0], ...], axis=0, keepdims=True)
    std = np.std(train[:, [0], ...], axis=0, keepdims=True)
    train_mask = (train[:, [1], ...] > 0.5).astype(np.uint8)
    val_mask = (val[:, [1], ...] > 0.5).astype(np.uint8)

    train_img = (train[:, [0], ...] - np.repeat(mean, train.shape[0], axis=0)) / np.repeat(std, train.shape[0], axis=0)
    val_img = (val[:, [0], ...] - np.repeat(mean, val.shape[0], axis=0)) / np.repeat(std, val.shape[0], axis=0)


    print('-'*30)
    print('Creating and compiling model...')
    model = UNet(args)
    best_path = os.path.join(args.save_dir, 'ckpt-best.hdf5')
    ckpt_best = ModelCheckpoint(best_path,
                                monitor='val_loss', save_best_only=True)

    est = EarlyStopping(monitor='val_loss', patience=3)
    print('-'*30)
    print('Fitting model...')

    datagen_train = MyGenerator(horizontal_flip_prob=0.,
                                vertical_flip_prob=0.,
                                elastic_alpha=2,
                                elastic_sigma=0.4,
                                affine_alpha=0.1)

    history = model.fit_generator(datagen_train.flow(train_img,
                                                     train_mask,
                                                     batch_size=args.batch_size,
                                                     shuffle=True),
                                  samples_per_epoch=len(train_img),
                                  validation_data=(val_img, val_img),
                                  verbose=1,
                                  nb_worker=4,
                                  nb_epoch=args.num_epochs,
                                  callbacks=[ckpt_best, est],
                                  pickle_safe=True)
    #
    # history = model.fit(train_img, train_mask,
    #                     validation_data=(val_img, val_mask),
    #                     batch_size=args.batch_size,
    #                     nb_epoch=args.num_epochs,
    #                     verbose=1, shuffle=True,
    #                     callbacks=[ckpt_best, est])

    print('Save history')
    path = os.path.join(args.save_dir, 'history.json')
    #
    with open(path, 'w') as fout:
        json.dump(history.history, fout)

    # print('-'*30)
    print('Save model structure data...')
    json_string = model.to_json()
    path = os.path.join(args.save_dir, 'model.json')
    with open(path, 'w') as fout:
        fout.write(json_string)
    #
    model.load_weights(best_path)

    test = np.load(args.test)
    test_img = (test[:, [0], ...] - np.repeat(mean, test.shape[0], axis=0)) / np.repeat(std, test.shape[0], axis=0)

    print('Predict test')
    mask = model.predict(test_img, batch_size=args.batch_size, verbose=1)
    np.save(os.path.join(args.save_dir, 'ssb_test_mask.npy'), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='path to data folder: raw and processed')
    parser.add_argument('--save_dir', type=str, default='../models/tmp',
                        help='directory to store checkpointed models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')

    parser.add_argument('--train', type=str,
                        default='../data/processed/ssb_raw_train.npy',
                        help='path to bulk of train imgs npy')
    parser.add_argument('--val', type=str,
                        default='../data/processed/ssb_raw_val.npy',
                        help='path to bulk of val imgs npy')
    parser.add_argument('--test', type=str,
                        default='../data/processed/x_raw_test.npy')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
