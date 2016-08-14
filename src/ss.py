# Let's seek for sequences

import numpy as np
import skimage
import os
import re
import matplotlib.pyplot as plt

from skimage.viewer import CollectionViewer


def main(path):
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

    return data


if __name__ == "__main__":
    train = '../data/raw/train'
    main(train)

