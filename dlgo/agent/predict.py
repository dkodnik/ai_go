import numpy as np

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil

class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        """Инициализация агента с помощью модели Keras и кодировщика доски."""
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        """Кодирование состояния доски."""
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)[0]

    def select_move(self, game_state):
        """Предсказание вероятностей ходов."""
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        #увеличение расстояния между наиболее вероятными и наименее вероятными ходами
        move_probs = move_probs ** 3
        eps = 1e-6
        #предотвращение слишком сильного приближения значения вероятности хода к 0 или 1
        move_probs = np.clip(move_probs, eps, 1-eps)
        #перенормировка с целью получения нового распределения вероятностей
        move_probs = move_probs / np.sum(move_probs)
        #преобразование вероятностей в ранжированный список ходов
        candidates = np.arange(num_moves)
        #отбор кандидатов на звание следующего хода
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            #последовательный перебор элементов списка с целью нахождения допустимого
            #хода, не приводящего к уменьшению глазного пространства
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                not is_point_an_eye(game_state.board, point, game_state.next_player):
                return goboard.Move.play(point)

        #при отсутствии допустимых и несамоубийственных вариантов ход пропускается
        return goboard.Move.pass_turn()

    def serialize(self, h5file):
        """Сериализация(сохранение) агента."""
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

def load_prediction_agent(h5file):
    #Десериализация агента из файла HDF5
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)