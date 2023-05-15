import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
from dlgo import kerasutil
import operator

class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        #узлы дерева имеют один родительский и в потенциале множество дочерних элементов
        self.parent = parent
        self.children = {}

        self.visit_count = 0
        self.q_value = 0
        #узел инициализируется значением априорной вероятности
        self.prior_value = probability
        #служебная функция будет обновлена во время поиска
        self.u_value = probability

    def select_child(self):
        """Выбор дочернего элемента AlphaGoNode путем максимизации Q-значения."""
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + \
                   child[1].u_value)

    def expand_children(self, moves, probabilities):
        """Добавление дочернего элемента"""
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(probability=prob)

    def update_values(self, leaf_value):
        """Обновление количества посещений, Q-значения и значения служебной функции узла AlphaGo"""
        if self.parent is not None:
            #сначала обновляются родительские узлы для гарантии того, что обход дерева осуществляется сверху вниз
            self.parent.update_values(leaf_value)

        #увеличение количества посещений данного узла на 1
        self.visit_count += 1

        #прибавление ценности указанного листа к Q-значению, нормализованному по количеству посещений
        self.q_value += leaf_value / self.visit_count

        if self.parent is not None:
            c_u = 5
            #обновление значения служебной функции с учетом текущего количества посещений
            self.u_value = c_u * np.sqrt(self.parent.visit_count) * self.prior_value / (1 + self.visit_count)

class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100):
        """Инициализация игрового агента AlphaGoMCTS."""
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()

    def select_move(self, game_state):
        """Выбор хода с помощью нейронной сети."""

        ## Основной метод алгоритма поиска по дереву в системе AlphaGo

        #симуляция заданного количества игр из текущего игрового состояния
        for simulation in range(self.num_simulations):
            current_state = game_state
            node = self.root
            #совершение ходов вплоть до достижения указанной глубины
            for depth in range(self.depth):
                #если текущий узел не имеет дочерних элементов...
                if not node.children:
                    if current_state.is_over():
                        break
                    #...добавьте их, используя вероятности, предоставленные сильной сетью политики
                    moves, probabilities = self.policy_probabilities(current_state)
                    node.expand_children(moves, probabilities)

                #если узел имеет дочерние элементы, вы можете выбрать один из них и совершить соответствующий ход
                move, node = node.select_child()
                current_state = current_state.apply_move(move)

                #вычисление выхода сети ценности и результата развертывания быстрой сети политики
                value = self.value.predict(current_state)
                rollout = self.policy_rollout(current_state)

                #определение значения комбинированной функции ценности
                weighted_value = (1 - self.lambda_value) * value + self.lambda_value * rollout

                #обновление значений этого узла при подъеме вверх по дереву
                node.update_values(weighted_value)

        ## Выбор наиболее посещаемого узла и обновление корневого узла дерева

        #в качестве следующего хода выберите наиболее посещаемый дочерний элемент корня
        move = max(self.root.children, key=lambda move: self.root.children.get(move).visit_count)

        self.root = AlphaGoNode()
        #если выбранный ход является дочерним элементом, задайте для него новый корневой узел
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None

        return move

    def serialize(self, h5file):
        raise IOError("Агент AlphaGoMCTS не может быть сериализован, " +
                      "рассмотрите возможность сериализации трех " +
                      "базовых нейронных сетей вместо этого.")

    def policy_probabilities(self, game_state):
        """Вычисление нормализованныех значений сильной сети политики для допустимых ходов.
        Этот метод возвращает как допустимые ходы, так и соответствующие им нормальзованные
        предсказания сети политики.
        """
        encoder = self.policy._encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        """Вычисление результата развертывания с использованием быстрой политики.
        Метод 'жадно' выбирает сильный ход в соответствии с быстрой политикой до
        доситижения предела развертывания, а затем определяет победителя. Он возвращает
        значения 1, если победил игрок, обладающий правом следующего хода, -1, если
        победил другой игрок, и 0, если не был достигнут ни один из результатов.
        """
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder
            valid_moves = [m for idx, m in enumerate(move_probabilities) if Move(encoder.decode_point_index(idx)) in game_state.legal_moves()]
            max_index, max_value = max(enumerate(valid_moves), key=operator.itemgetter(1))
            max_point = encoder.decode_point_index(max_index)
            greedy_move = Move(max_point)
            if greedy_move in game_state.legal_moves():
                game_state = game_state.apply_move(greedy_move)

        next_player = game_state.next_player
        winner = game_state.winner()
        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0
