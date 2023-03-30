from __future__ import absolute_import
import tempfile
import os

import h5py
import keras
from keras.models import load_model, save_model


def save_model_to_hdf5_group(model, f):
    # Используется Keras save_model для сохранения полной
    # модели (включая состояние оптимизатора) в файл.
    # Затем мы можем встроить содержимое этого файла HDF5 в наш.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        save_model(model, tempfname)
        serialized_model = h5py.File(tempfname, 'r')
        root_item = serialized_model.get('/')
        serialized_model.copy(root_item, f, 'kerasmodel')
        serialized_model.close()
    finally:
        os.unlink(tempfname)


def load_model_from_hdf5_group(f, custom_objects=None):
    # Распаковка модели во временный файл. Затем мы можем
    # использовать Keras load_model для его чтения.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        serialized_model = h5py.File(tempfname, 'w')
        root_item = f.get('kerasmodel')
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value
        for k in root_item.keys():
            f.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()
        return load_model(tempfname, custom_objects=custom_objects)
    finally:
        os.unlink(tempfname)


def set_gpu_memory_target(frac):
    """Настройте Tensorflow так, чтобы он использовал часть доступной памяти графического
    процессора (GPU).

    Используйте это для параллельной оценки моделей. По умолчанию Tensorflow попытается
    заранее сопоставить всю доступную память графического процессора (GPU). Вы можете
    настроить использование только части, чтобы несколько процессов могли выполняться
    параллельно. Например, если вы хотите использовать 2 воркера, установите долю памяти
    равной 0,5.

    Если вы используете многопроцессорную обработку Python, вы должны вызвать эту функцию
    из *воркера* процесса (не из родительского).

    Эта функция ничего не делает, если Keras использует в качестве бэкенда не фреймворк Tensorflow.
    """
    if keras.backend.backend() != 'tensorflow':
        return
    # Выполняется импорт здесь, а не вверху, на случай,
    # если Tensorflow вообще не установлен.
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    set_session(tf.Session(config=config))
