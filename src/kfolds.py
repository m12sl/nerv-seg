import numpy as np
import os

def main():
    try:
        os.makedirs('../data/processed/fold_0')
    except Exception as err:
        print(err)

    idx = np.load('../data/processed/train_idx_.npy')
    
    data = np.load('../data/processed/train_img_80x64.npy')
    target = np.load('../data/processed/train_mask_80x64.npy')
    
    subjects = np.array(list(set(idx[:, 0])))
    np.random.seed(2016)
    np.random.shuffle(subjects)

    train_subj = subjects[6:]
    val_subj = subjects[:6]

    train_idx = np.concatenate([np.where(idx[:, 0] == s)[0] for s in train_subj])
    val_idx = np.concatenate([np.where(idx[:, 0] == s)[0] for s in val_subj])


    img_train = data[train_idx]
    mask_train = target[train_idx]

    img_val = data[val_idx]
    mask_val = target[val_idx]

    np.save('../data/processed/fold_0/train_img_.npy', img_train)
    np.save('../data/processed/fold_0/train_mask_.npy', mask_train)
    np.save('../data/processed/fold_0/val_img_.npy', img_val)
    np.save('../data/processed/fold_0/val_mask_.npy', mask_val)


if __name__ == "__main__":
    main()


