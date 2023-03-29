import os.path
import tarfile
import gzip
import glob
import shutil

import numpy as np
from keras.utils import to_categorical

from dlgo.gosgf import Sgf_game
from dlgo.goboard import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name

from dlgo.data.index_processor import KGSIndex
#Для создания набора обучающих и тестовых данных будет использовать сэмплер Sampler
from dlgo.data.sampling import Sampler

class GoDataProcessor:
    def __init__(self, encoder='oneplane', data_directory='data'):
        """"Инициализация обработчика данных.
        Указание кодировщика и локальной директории с данными.
        """
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(self, data_type='train', num_samples=1000):
        """Загрузка, обработка и сохранение данных.
        Параметры:
            data_type - может быть 'train' - обучающие данные или 'test' - тестовые данные
            num_samples - задаёт количество партий, данные которые требуется загрузить
        """
        index = KGSIndex(data_directory=self.data_dir)
        # загрузка игр с сервера KGS, если уже загружены - повторно грузится не будут
        index.download_files()

        #разделение партий на уровне файлов! чтобы ходы одной из партий полностью входили
        #в заданный набор данных определенного типа train или test
        sampler = Sampler(data_dir=self.data_dir)
        #отбирает указанное количество партий для формирования набора данных заданного типа
        data = sampler.draw_data(data_type, num_samples)

        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in data:
            #составление списка всех имен zip-файлов
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            #группировка всех индексов SGF-файлов по имени zip-файла
            indices_by_zip_name[filename].append(index)
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                #обработка отдельных zip-файлов
                self.process_zip(zip_name, data_file_name, indices_by_zip_name[zip_name])

        #объединение и возврат признаков и меток из каждого zip-файла
        features_and_labels = self.consolidate_games(data_type, data)
        return features_and_labels

    def unzip_data(self, zip_file_name):
        # Распаковка файла `gz` в файл `tar`.
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)
        # Удалить ".gz" в конце, чтобы получить имя файла 'tar'.
        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')
        # Скопировать содержимое распакованного файла в файл `tar`.
        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        """Преобразование записей партий игры Го, хранящихся в zip-файлах,
        в закодированные признаки и метки
        """
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        #определение общего количества ходов во всех партиях, содержащихся в данном zip-файле
        total_examples = self.num_total_examples(zip_file, game_list, name_list)
        #определение формы признаков и меток на основе используемого кодировщика
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))
        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            #чтение содержимого файла SGF в виде строки после распаковки zip-файла
            sgf = Sgf_game.from_string(sgf_content)

            #определение начального игрового состояния путем применения
            #всех камней "гандикапа"
            game_state, first_move_done = self.get_handicap(sgf)

            #итеративная обработка всех ходов в файле SGF
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        #считывание координат камня, который предстоит поместеть на доску
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        #...или пропуск хода при отсутствии такого камня
                        move = Move.pass_turn()

                    if first_move_done and point is not None:
                        #кодирование текущего игрового состояния в виде признаков...
                        features[counter] = self.encoder.encode(game_state)
                        #...а следующего хода - в виде метки для этих признаков
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    #применение хода к доске и повторение цикла для следующего хода
                    game_state = game_state.apply_move(move)
                    first_move_done = True
        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'
        chunk = 0 # Из-за файлов с большим содержимым разделяются после chunksize
        chunksize = 1024
        #обработка признаков и меток в виде фрагментов размером 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            #текущий фрагмент отсекается от признаков и меток...
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            #...и сохраняется в отдельном файле
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def num_total_examples(self, zip_file, game_list, name_list):
        """Вычисление общего количества ходов, доступных в текущем zip-файле.
        Позволяет определить размер массивов признаков и меток.
        """
        total_examples = 0
        for index in game_list:
            name = name_list[index+1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

    @staticmethod
    def get_handicap(sgf):
        """Определение количества камней гандикапа и их применение к пустой доске Го."""
        board_size = 19
        go_board = Board(board_size, board_size)
        first_move_done = False
        move = None
        game_state = GameState.new_game(board_size)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row+1, col+1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def consolidate_games(self, data_type, samples):
        """Объединение массивов признаков и меток NumPy в один набор.
        Реализация метода конкатенации.
        """
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)

        features_list = []
        label_list = []
        for file_name in file_names:
            file_prifix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prifix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19*19)
                features_list.append(x)
                label_list.append(y)
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        np.save('{}/features_{}.npy'.format(self.data_dir, data_type), features)
        np.save('{}/labels_{}.npy'.format(self.data_dir, data_type), labels)

        return features, labels