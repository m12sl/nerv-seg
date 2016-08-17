from __future__ import print_function

import numpy as np

import argparse
import os
import errno
import json

from model import UNet, DNet
from keras.callbacks import ModelCheckpoint
from generator import MyGenerator


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

    maybe_mkdir(args.res)
    args.res = args.res + 'fold-{}'.format(args.fold)

    print('-'*30)
    print('Creating and compiling model...')
    if args.model == 'dnet':
        model = DNet(args)
    else:
        model = UNet(args)
    best_path = os.path.join(args.save_dir, 'ckpt-best.hdf5')
    ckpt_best = ModelCheckpoint(best_path,
                                monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Loading and preprocessing train data...')
    img = np.load(args.bulk)
    mask = np.load(args.bulk.replace('_img_', '_mask_'))

    #train_index = np.load(args.index)
    #val_index = np.load(args.val_index)
    index = np.arange(len(img))
    np.random.seed(442)
    np.random.shuffle(index)
    train_index = index[:4800]
    val_index = index[4800:]
    
    img_train = img[train_index, ...]
    img_val = img[val_index, ...]
    mask_train = mask[train_index, ...]
    mask_val = mask[val_index, ...]
    print("Train shape:", img_train.shape, ", valid shape: ", img_val.shape)

    datagen_train = MyGenerator(horizontal_flip_prob=0., 
                                vertical_flip_prob=0., 
                                elastic_alpha=2, 
                                elastic_sigma=0.4, 
                                affine_alpha=0.1)

    print('-'*30)
    print('Fitting model...')
    history = model.fit_generator(datagen_train.flow(img_train,
                                                     mask_train,
                                                     batch_size=args.batch_size,
                                                     shuffle=True),
                                  samples_per_epoch=len(img_train), 
                                  validation_data=(img_val, mask_val),
                                  verbose=2,
                                  nb_worker=1,
                                  nb_epoch=args.num_epochs,
                                  callbacks=[ckpt_best],
                                  pickle_safe=False)

    '''
    history = model.fit(img_train, mask_train,
                        validation_data=(img_val, mask_val),
                        batch_size=args.batch_size,
                        nb_epoch=args.num_epochs,
                        verbose=1, shuffle=True,
                        callbacks=[ckpt_best])
    '''

    print('Save history')
    path = os.path.join(args.save_dir, 'history.json')

    with open(path, 'w') as fout:
        json.dump(history.history, fout)

    print('-'*30)
    print('Save model structure data...')
    json_string = model.to_json()
    path = os.path.join(args.save_dir, 'model.json')
    with open(path, 'w') as fout:
        fout.write(json_string)

    model.load_weights(best_path)

    print('-'*30)
    print('Predicting masks on all data')
    # lets predicts the whole train (train, val, test) + test

    mask_train = model.predict(img, verbose=1)
    np.save(args.res + 'train.npy', mask_train)

    test = np.load(args.test)
    mask = model.predict(test, verbose=1)
    np.save(args.res + 'test.npy', mask)


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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers during model fit')
    # yes, masks path will be compute in dataloader
    parser.add_argument('--fold', type=int, default=0,
                        help='number of fold to train')
    parser.add_argument('--bulk', type=str, 
                        default='../data/processed/train_img_80x64.npy',
                        help='path to bulk of train imgs npy')
    parser.add_argument('--test', type=str,
                        default='../data/processed/test_img_80x64.npy')
    # parser.add_argument('--index', type=str,
    #                     default='../data/processed/folds/train-{}.npy',
    #                     help='path to bulk of val imgs npy')
    # parser.add_argument('--val_index', type=str,
    #                     default='../data/processed/folds/val-{}.npy')
    parser.add_argument('--model', type=str,
                        default='unet')
    parser.add_argument('--res', type=str,
                        default='../data/predict-0/')
    args = parser.parse_args()

    assert args.fold in range(0, 9), "Wrong fold specified"
    args.index = '../data/processed/folds/train-{}.npy'.format(args.fold)
    args.val_index = '../data/processed/folds/val-{}.npy'.format(args.fold)

    train(args)


if __name__ == '__main__':
    main()
