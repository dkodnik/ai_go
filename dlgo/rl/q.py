import numpy as np

from keras.optimizers import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import goboard

class QAgent(Agent):
    """Агент применяющий алгоритм Q-обучения -> 'Ценность действия'."""

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

    def set_temperature(self, temperature):
        #параметр temperature - это значение Е, определяющее степень случайности политики
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        """Выбор ходов для агента Q-обучения, с реализующий Е-жадную политику."""

        board_tensor = self.encoder.encode(game_state)

        #генерация списка всех допустимых ходов.
        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor)

        #если допустимых ходов не осталось, агент может пропустить ход
        if not moves:
            return goboard.Move.pass_turn()

        num_moves = len(moves)
        board_tensors = np.array(board_tensors)
        #унитарное кодирование всех допустимых ходов
        move_vectors = np.zeros(
            (num_moves, self.encoder.num_points()))
        for i, move in enumerate(moves):
            move_vectors[i][move] = 1

        #это форма функции predict с двумя входами: передаем эти два входа в виде списка
        values = self.model.predict([board_tensors, move_vectors])
        values = values.reshape(len(moves)) #значения будут представлены в виде матрицы N*1, где N-кол. допустимых ходов. Вызов функции reshape преобразует её в вектор размером N

        #ранжирование ходов в соответствии с Е-жадной политикой
        ranked_moves = self.rank_moves_eps_greedy(values)

        #выбор первого несамоубийственного хода из списка, как в случае игры агента с самим собой
        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                #запись решения в буфер данных опыта
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=moves[move_idx],
                    )
                    return goboard.Move.play(point)
            #этого этапа вы достигните в том случае, если все допустимые ходы будут признаны самоубийственными
            return goboard.Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        """Сортировка ходов в порядке уменьшения ценности."""
        #в случае исследования ходы упорядочиваются случайным образом, а не в соответствии с реальными значениями
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        #получение индексов ходов в порядке возрастания ценности
        ranked_moves = np.argsoft(values)
        #[::1]-это самый эффективный способ обратить порядок элементов вектора в NumPy. В
        #результате ходы будут ранжированы в порядке уменьшения ценности.
        return ranked_moves[::-1]

    def train(self, experience, lr=0.1, batch_size=128):
        """Обучение агента Q-обучения на основе набранного им опыта.
        lr и batch_size - это параметры, позволяющие точно настроить процесс обучения.
        """
        opt = SGD(lr=lr) # СГС
        #mse - это среднеквадратичная ошибка, используется вместо categorical_crossentropy, потому что имеем дело с непрерывной функцией
        self.model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            actions[i][action] = 1
            y[i] = reward
        self.model.fit(
            [experience.states, actions], y, #передаёт два разных входа в виде списка
            batch_size=batch_size,
            epochs=1)