from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D

def alphago_model(input_shape, is_policy_net=False,
                  num_filters=192,
                  first_kernel_size=5,
                  other_kernel_size=3):
    """Инициализация нейронной сети для сетей политики и ценности AlphaGo.

    Params:
        input_shape - ...
        is_policy_net - это логический флаг позволяет указать тип нужной сети (сети политики или сети ценности)
        num_filters - все сверточные слои, кроме последнего, имеют одинаковое количество фильтров
        first_kernel_size - первый слой ядра, имеет размер по умолчанию 5
        other_kernel_size - все остальные ядра, имеют размер по умолчанию 3
    """
    model = Sequential()
    model.add(
        Conv2D(num_filters, first_kernel_size, input_shape=input_shape,
               padding='same', data_format='channels_first', activation='relu'))
    for i in range(2, 12):
        #первые 12 слоев сети политики и сети ценности программы AlphaGo являются одинаковыми
        model.add(Conv2D(num_filters, other_kernel_size, padding='same',
                         data_format='channels_first', activation='relu'))

    if is_policy_net:
        # Создание сильной сети политики AlphaGo в Keras
        model.add(Conv2D(filters=1, kernel_size=1, padding='same',
                         data_format='channels_first', activation='softmax'))
        model.add(Flatten())
        return model

    else:
        # Построение сети ценности AlphaGo в Keras
        model.add(Conv2D(num_filters, other_kernel_size, padding='same',
                         data_format='channels_first', activation='relu'))
        model.add(Conv2D(filters=1, kernel_size=1, padding='same',
                         data_format='channels_first', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model