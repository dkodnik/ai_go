"""Модель небольшой сверточной сети для предсказания ходов в игре Го"""
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D

def layers(input_shape):
    return [
        # дополнение нулями используется для увеличения размера входных изображений
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        Conv2D(48, (7,7), data_format='channels_first'),
        Activation('relu'),

        # порядок channel_first говорит о том, что первым будет указан размер
        # вх.плоскости признаков
        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]