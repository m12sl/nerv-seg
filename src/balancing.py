# let's create 2 train/val sets:
# only nice marked: from 30 to 90 masks in set
# stratified by subjects and random
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import multiprocessing as mp
import gc

ROWS = 420
COLS = 580

BROWS = 128
BCOLS = 160


def check_mask(path):
    img = imread(path[:-4] + '_mask.tif')
    return np.sum(img) > 0


def get_mask_statistics(path='../data/raw/train/'):
    import os
    import re
    files = os.listdir(path)
    data = {}
    for f in files:
        r = re.match('(\d+)_(\d+).tif', f)
        if r:
            subj = int(r.group(1))
            no = int(r.group(2))
            if subj not in data:
                data[subj] = []
            data[subj].append((no, os.path.join(path, f)))
    for x in data.keys():
        data[x] = sorted(data[x], key=lambda t: t[0])

    tasks = []
    for subj in data.keys():
        for x in data[subj]:
            name = x[1]
            tasks.append((subj, x[0], name))

    print(len(tasks))
    ret = [(subj, no, name, check_mask(name)) for (subj, no, name) in tqdm(tasks)]

    table = {}
    for (subj, no, name, has_mask) in ret:
        if subj not in table:
            # masked is number of images with masks, pos contains their names
            table[subj] = {'pos': [], 'neg': [], 'masked': 0, 'total': 0}

        table[subj]['total'] += 1
        if has_mask:
            table[subj]['pos'].append(name)
            table[subj]['masked'] += 1
        else:
            table[subj]['neg'].append(name)

    return table


def subj_stratified(table, seed=19231, with_unmasked=True, extreme=False):
    def pack(keys, table, flag):
        ret = []
        for k in keys:
            lst = table[k]['pos']
            if flag:
                lst = lst + table[k]['neg']
            for x in lst:
                img = x
                mask = x[:-4] + '_mask.tif'
                ret.append((img, mask))
        return ret

    nice = {_: table[_] for _ in table.keys() if (30 <= table[_]['masked'] <= 90)}
    print('There are {} series appropriate by markup'.format(len(nice)))

    keys = sorted(nice.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)
    N = 5
    train_keys = keys[N:]
    val_keys = keys[:N]

    train_list = pack(train_keys, table, with_unmasked)
    val_list = pack(val_keys, table, with_unmasked)

    return train_list, val_list


def random_stratified(table, seed=19231, with_unmasked=True):
    nice = [table[_] for _ in table.keys() if (30 <= table[_]['masked'] <= 90)]
    print('There are {} series appropriate by markup'.format(len(nice)))

    files = []
    for x in nice:
        files.extend(x['pos'])
        if with_unmasked:
            files.extend(x['neg'])

    # just for machine invariance, mb wrong useless
    files = sorted(files)
    files = [(x, x[:-4] + '_mask.tif') for x in files]

    np.random.seed(seed)
    np.random.shuffle(files)

    N = int(0.2 * len(files))
    train_list = files[N:]
    val_list = files[:N]

    return train_list, val_list


def extreme_random_stratified(table, seed=19231):
    files = []
    for subj, x in table.items():
        if x['masked'] < 60:
            files.extend(x['pos'])
        if x['masked'] > 60:
            files.extend(x['neg'])

    files = sorted(files)
    files = [(x, x[:-4] + '_mask.tif') for x in files]

    np.random.seed(seed)
    np.random.shuffle(files)

    N = int(0.2 * len(files))

    train_list = files[N:]
    val_list = files[:N]

    return train_list, val_list



def test_files(path='../data/raw/test/'):
    import os
    import re
    files = os.listdir(path)
    test_list = []
    for f in files:
        r = re.match('(\d+).tif', f)
        if r:
            no = int(r.group(1))
            test_list.append((no, os.path.join(path, f)))
    test_list.sort(key=lambda t: t[0])

    return [t[1] for t in test_list]



def func(task):
    no = task[0]
    im = resize(imread(task[1][0]), (BROWS, BCOLS)).astype(np.float32)
    mask = resize(imread(task[1][1]), (BROWS, BCOLS)) > 0.5
    return (no, im, mask)


def func_img_only(task):
    no = task[0]
    im = resize(imread(task[1]), (BROWS, BCOLS)).astype(np.float32)
    return (no, im)


def build_image_bulk(list_of_files):
    bulk = np.zeros((len(list_of_files), 2, BROWS, BCOLS))

    print('Read files')
    tasks = enumerate(list_of_files)
    it = tqdm(tasks, total=len(list_of_files))
    pool = mp.Pool(processes=4)
    work = pool.imap_unordered(func, it)
    pool.close()
    pool.join()

    print('Merge {} results'.format(len(list_of_files)))
    for (i, x, y) in tqdm(work):
        bulk[i, 0, ...] = x[...]
        bulk[i, 1, ...] = y[...]
    return bulk


def build_test_bulk(list_of_files):
    bulk = np.zeros((len(list_of_files), 1, BROWS, BCOLS))

    tasks = enumerate(list_of_files)
    it = tqdm(tasks, total=len(list_of_files))
    pool = mp.Pool(processes=4)
    work = pool.imap_unordered(func_img_only, it)
    pool.close()
    pool.join()

    print('Merge {} results'.format(len(list_of_files)))
    for (i, x) in tqdm(work):
        bulk[i, 0, ...] = x[...]
    return bulk



if __name__ == "__main__":
    table = get_mask_statistics()
    # lst = subj_stratified(table)
    lst = extreme_random_stratified(table)

    print('Build TRAIN bulk')
    train = build_image_bulk(lst[0])
    np.save('../data/processed/erb_raw_train.npy', train)
    del train

    gc.collect()
    print('Build VAL bulk')
    val = build_image_bulk(lst[1])
    np.save('../data/processed/erb_raw_val.npy', val)
    del val
    gc.collect()


    # print('Build TRAIN bulk')
    # train = build_image_bulk(lst[0])
    # np.save('../data/processed/ssb_raw_train.npy', train)
    # del train
    #
    # gc.collect()
    # print('Build VAL bulk')
    # val = build_image_bulk(lst[1])
    # np.save('../data/processed/ssb_raw_val.npy', val)
    # del val
    # gc.collect()
    #
    # lst = random_stratified(table)
    # print('Build TRAIN bulk')
    # train = build_image_bulk(lst[0])
    # np.save('../data/processed/rsb_raw_train.npy', train)
    # del train
    # gc.collect()
    #
    # print('Build VAL bulk')
    # val = build_image_bulk(lst[1])
    # np.save('../data/processed/rsb_raw_val.npy', val)
    # del val
    # gc.collect()
    #
    # lst = test_files()
    # print(lst)
    # test = build_test_bulk(lst)
    # np.save('../data/processed/x_raw_test.npy', test)

