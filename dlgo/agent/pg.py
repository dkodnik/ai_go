"""Градиентное обучение агента политики (Policy)"""
import numpy as np

from keras import backend as K
from keras.optimizers import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil

class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        #экземпляр последовательной модели Keras
        self._model = model
        #реализация интерфейса кодировщика
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0

    def serialize(self, h5file):
        """Сериализация(сохранение) агента"""
        h5file.create_group('encoder')

        #Сохранение достаточного количества информации для
        # воссоздания кодировщика доски.
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._endcoder.board_width
        h5file['encoder'].attrs['board_height'] = self._endcoder.board_height

        h5file.create_group('model')
        #использование встроенных функций Keras для сохранения модели и весов
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def select_move(self, game_state):
        """Выбор хода с помощью нейронной сети."""
        #!код немного не соответствует что в книге а что в коде на github

        # создание массива с индексами всех точек доски
        num_moves = self._encoder.board_width * self._encoder.board_height

        #состояние доски кодируется в виде тензора
        board_tensor = self._encoder.encode(game_state)

        #вызов функции predict библиотеки Keras производит пакетные предсказания,
        #поэтому мы преобразуем отдельное состояние доски в массив и извлекаем из
        #него первый элемент
        x = np.array([board_tensor])

        #! move_probs = clip_probs(move_probs) <--- такого нет
        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            move_probs = self._model.predict(x)[0]
        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        #перенормировка с целью получения нового распределения вероятностей
        move_probs = move_probs / np.sum(move_probs)

        # преобразование вероятностей в ранжированный список ходов
        candidates = np.arange(num_moves)

        #сэмплирование из точек доски в соответствии с политикой, создание
        #ранжированного списка точек для выполнения попыток выбора хода
        ranked_moves = np.random.choice(
            candidates, num_moves,
            replace=False, p=move_probs)

        #циклический перебор точек, проверка хода на допустимость
        #и выбор первого из таких ходов
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(
                game_state.board,
                point,
                game_state.next_player)
            if is_valid and (not is_an_eye):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=board_tensor,
                        action=point_idx)
                return goboard.Move.play(point)
        #достижение этого этапа означает, что разумных ходов больше нет
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512):
        opt = SGD(lr=lr, clipnorm=clipnorm)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt)

        n = experience.states.shape[0]
        # Translate the actions/rewards.
        num_moves = self._encoder.board_width * self._encoder.board_height
        y = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1)

    def predict(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self._model.predict(input_tensor)[0]

    def set_temperature(self, temperature):
        self._temperature = temperature

    def set_collector(self, collector):
        self._collector = collector

def load_policy_agent(h5file):
    """Загрузка агента политики из файла"""

    #использование встроенных функций Keras для загрузки структуры и весов модели
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    #восстановление кодировщика состояния доски
    encoder_name = h5file['encoder'].attrs['name']
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    #воссоздание агента
    return PolicyAgent(model, encoder)