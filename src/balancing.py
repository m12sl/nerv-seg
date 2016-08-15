# let's create 2 train/val sets:
# only nice marked: from 30 to 90 masks in set
# stratified by subjects and random
import numpy as np


def check_mask(path):
    from skimage.io import imread
    img = imread(path)
    return np.sum(img) > 0


def get_mask_statistics(path='../data/raw/train/'):
    import os
    import re
    from tqdm import tqdm
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
            name = x[1][:-4] + '_mask.tif'
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


def subj_stratified(table, seed=19231, with_unmasked=True):
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


if __name__ == "__main__":
    table = get_mask_statistics()
    subj_stratified(table)
    random_stratified(table)
