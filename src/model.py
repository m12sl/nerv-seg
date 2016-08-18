from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import AtrousConvolution2D, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam

# from balancing import BROWS, BCOLS
BROWS = 128
BCOLS = 160


from keras import backend as K

SMOOTH = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def dice_coef2(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + SMOOTH
    union = K.sum(y_true_f, axis=1, keepdims=True) + \
            K.sum(y_pred_f, axis=1, keepdims=True) + \
            SMOOTH
    return K.mean(intersection / union)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef2(y_true, y_pred)


def UNet(args):
    inputs = Input((1, BROWS, BCOLS))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)


    up5 = UpSampling2D(size=(2, 2))(conv5)

    up6 = merge([up5, conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Convolution2D(1, 1, 1)(conv9)
    dropout = Dropout(0.1)(conv10)
    output = Activation('sigmoid')(dropout)

    model = Model(input=inputs, output=output)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef2])

    return model


def DNet(args):
    inputs = Input((1, args.img_height, args.img_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)

    conv2 = AtrousConvolution2D(64, 3, 3, atrous_rate=(2, 2), activation='relu', border_mode='same')(conv1)
    conv2 = AtrousConvolution2D(64, 3, 3, atrous_rate=(2, 2), activation='relu', border_mode='same')(conv2)

    conv3 = AtrousConvolution2D(64, 3, 3, atrous_rate=(4, 4), activation='relu', border_mode='same')(conv2)
    conv3 = AtrousConvolution2D(64, 3, 3, atrous_rate=(4, 4), activation='relu', border_mode='same')(conv3)

    conv4 = AtrousConvolution2D(64, 3, 3, atrous_rate=(8, 8), activation='relu', border_mode='same')(conv3)
    conv4 = AtrousConvolution2D(64, 3, 3, atrous_rate=(8, 8), activation='relu', border_mode='same')(conv4)

    merge4 = merge([conv4, conv3, conv2, conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(merge4)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    conv6 = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=inputs, output=conv6)
    model.compile(optimizer=Adam(lr=1e-2), loss=dice_coef_loss, metrics=[dice_coef2])

    return model
