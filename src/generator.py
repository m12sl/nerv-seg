import numpy as np
import cv2


class MyGenerator():
    def __init__(self, horizontal_flip_prob=0., vertical_flip_prob=0., elastic_alpha=None, elastic_sigma=None,
                 affine_alpha=None):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.affine_alpha = affine_alpha
        self.CHANNEL_INDEX = 1
        self.ROW_INDEX = 2
        self.COL_INDEX = 3

    def _make_batches(self, batch_size, size):
        nb_batch = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(0, nb_batch)]

    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def elastic_transform(self, x, y):
        x, y = x[0, :, :], y[0, :, :]
        shape_size = x.shape

        alpha = x.shape[1] * self.elastic_alpha
        sigma = x.shape[1] * self.elastic_sigma
        alpha_affine = x.shape[1] * self.affine_alpha
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, \
                           [center_square[0] + square_size, center_square[1] - square_size], \
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape) \
            .astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        x = cv2.warpAffine(x, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        y = cv2.warpAffine(y, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        # Elastic
        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), \
                              ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), \
                              ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        xx, yy = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        xx = np.reshape(xx + dx, (-1, 1)).astype('float32')
        yy = np.reshape(yy + dy, (-1, 1)).astype('float32')

        return (cv2.remap(x, xx, yy, \
                          interpolation=cv2.INTER_LINEAR, \
                          borderMode=cv2.BORDER_REFLECT).reshape(shape_size)[np.newaxis],
                cv2.remap(y, xx, yy, \
                          interpolation=cv2.INTER_LINEAR, \
                          borderMode=cv2.BORDER_REFLECT).reshape(shape_size)[np.newaxis])

    def transform(self, x, y):
        img_col_index = self.COL_INDEX - 1
        img_row_index = self.ROW_INDEX - 1

        if np.random.random() < self.horizontal_flip_prob:
            x = self.flip_axis(x, img_col_index)
            y = self.flip_axis(y, img_col_index)

        if np.random.random() < self.vertical_flip_prob:
            x = self.flip_axis(x, img_row_index)
            y = self.flip_axis(y, img_row_index)

        if self.elastic_alpha or self.elastic_sigma or self.affine_alpha:
            x, y = self.elastic_transform(x, y)
        return x, y

    def flow(self, img, mask, batch_size, shuffle=False):
        assert img.shape == mask.shape
        size = len(mask)
        while 1:
            batches = self._make_batches(batch_size, size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                current_batch_size = batch_end - batch_start
                if batch_index % 50 == 0:
                    print("Processing %i-th batch." % batch_index)
                indeces = np.arange(batch_start, batch_end)
                if shuffle:
                    np.random.shuffle(indeces)

                img_batch = np.zeros(tuple([current_batch_size] + list(img.shape)[1:]))
                mask_batch = np.zeros(tuple([current_batch_size] + list(mask.shape)[1:]))
                for i, (x, y) in enumerate(zip(img[indeces], mask[indeces])):
                    img_batch[i], mask_batch[i] = self.transform(x, y)

                yield img_batch, mask_batch
