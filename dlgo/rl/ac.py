import numpy as np

from keras.optimizers import SGD

from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye

class ACAgent(Agent):
    """Агент 'Актор-Критик'."""

    def __init__(self, model, encoder):
        #экземпляр последовательной модели Keras
        self.model = model
        #реализация интерфейса кодировщика
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        #позволяет драйверу игры бота с самим собой прикрепить к агенту объект ExperienceCollector
        self.collector = collector

    def select_move(self, game_state):
        """Выбор хода с помощью нейронной сети."""

        # создание массива с индексами всех точек доски
        num_moves = self.encoder.board_width * self.encoder.board_height

        # состояние доски кодируется в виде тензора
        board_tensor = self.encoder.encode(game_state)

        # вызов функции predict библиотеки Keras производит пакетные предсказания,
        # поэтому мы преобразуем отдельное состояние доски в массив и извлекаем из
        # него первый элемент
        x = np.array([board_tensor])

        #поскольку эта модель предусматривает два выхода, функция predict возвращает
        #кортеж с двумя массивами NumPy
        actions, values = self.model.predict(x)

        #вызов функции predict - это вызов пакета, позволяющий одновременно обрабатывать
        #несколько состояний доски, поэтому для получения желаемого распределения вероятностей
        #вам необходимо выбрать первый элемент массива
        move_probs = actions[0]

        #значения представлены в виде одномерного вектора, поэтому для получения значения
        #в виде простого числа с плавающей запятой вы должны извлечь первый элемент
        estimated_value = values[0][0]

        # предотвращение слишком сильного приближения значения вероятности хода к 0 или 1
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # перенормировка с целью получения нового распределения вероятностей
        move_probs = move_probs / np.sum(move_probs)

        # преобразование вероятностей в ранжированный список ходов
        candidates = np.arange(num_moves)

        # сэмплирование из точек доски в соответствии с политикой, создание
        # ранжированного списка точек для выполнения попыток выбора хода
        ranked_moves = np.random.choice(
            candidates, num_moves,
            replace=False, p=move_probs)

        # циклический перебор точек, проверка хода на допустимость
        # и выбор первого из таких ходов
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye( #fills own eye - заполняет собственный глаз
                game_state.board,
                point,
                game_state.next_player)
            if move_is_valid and (not is_an_eye):
                # при выборе хода уведомляет ExperienceCollector о принятом решении
                if self.collector is not None:
                    #включение оценочного значения в буфер с данными опыта
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value)
                return goboard.Move.play(point)
        # достижение этого этапа означает, что разумных ходов больше нет
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.1, batch_size=128):
        """Обучение
        Params:
            lr - скорость обучения
            batch_size - определяет кол.ходов из данных опыта, учитываемых при обновлении отдельного веса
        """
        # используется стохастический градиентный спуск
        opt = SGD(lr=lr, clipvalue=0.2)
        self.model.compile(
            optimizer=opt,
            #categorical_crossentropy - функция потери для выхода политики.
            #mse (среднеквадратичная ошибка) - функция потери для выхода ценности.
            #Порядок соответствует порядку в конструкторе Model
            loss=['categorical_crossentropy', 'mse'],
            #к выходу полтитики применен вес 1.0, а к выходу ценности 0.5
            loss_weights=[1.0, 0.5],
        )

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))
        for i in range(n):
            #эта схема кодирования аналогичка схеме "политики", но взвешена с учетом преимущества
            action = experience.actions[i]
            policy_target[i][action] = experience.advantages[i]

            #эта схема кодирования аналогичка схеме - "ценности"
            reward = experience.rewards[i]
            value_target[i] = reward

        self.model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)

    def serialize(self, h5file):
        """Сериализация(сохранение) агента"""
        h5file.create_group('encoder')

        # Сохранение достаточного количества информации для
        # воссоздания кодировщика доски.
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height

        h5file.create_group('model')
        # использование встроенных функций Keras для сохранения модели и весов
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
    """Загрузка агента политики из файла"""

    #использование встроенных функций Keras для загрузки структуры и весов модели
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    #восстановление кодировщика состояния доски
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    #воссоздание агента
    return ACAgent(model, encoder)
