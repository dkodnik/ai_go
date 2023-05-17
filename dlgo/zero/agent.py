import numpy as np
from keras.optimizers import SGD

from ..agent import Agent

__all__ = [
    'ZeroAgent',
]


class Branch:
    def __init__(self, prior):
        """Структура для отслеживания статистики ветви."""
        self.prior = prior #предварительный|предыдущий
        self.visit_count = 0 #количество посещений ветви
        self.total_value = 0.0 #общее значение

class ZeroTreeNode:
    """Узел дерева поиска в стиле AlphaGo Zero (AGZ)."""
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        #в корне дерева параметры parent и last_move будут иметь значение None
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        #позднее дочерние элементы будут отображены из move в другой ZeroTreeNode
        self.children = {}

    def moves(self):
        """Возвращает список всех возможных ходов из данного узла."""
        return self.branches.keys()

    def add_child(self, move, child_node):
        """Позволяет добавить в дерево новые узлы."""
        self.children[move] = child_node

    def has_child(self, move):
        """Проверяет наличие дочернего узла для конкретного хода."""
        return move in self.children

    def get_child(self, move):
        """Возвращает определенный дочерний узел."""
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        """Выбор дочерней ветви."""
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n+1)
        #node.moves() - это список ходов. При передаче key=score_branch
        #функция max возвращает ход с наибольшим значением функции score_branch
        return max(node.moves(), key=score_branch)

    def select_move(self, game_state):
        """Выбор хода с помощью нейронной сети."""

        ## Спуск по дереву поиска:
        #см.ниже реализацию функции create_node
        root = self.create_node(game_state)

        #это первый этап процесса, который повторяется много раз для
        #каждого хода. self.num_moves определяет количество повторений
        #цикла поиска.
        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                #если функция has_child возвращает значение False,
                #значит, мы достигли концевого узла дерева
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, parent=node) #child_node = self.create_node(new_state, move=next_move, parent=node)

            move = next_move
            #на каждом уровне дерева мы переключаем перспективу между двумя
            #игроками, что требует умножения значения на -1:то, что хорошо для
            #черных, плохо для белых, и наоборот
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

        #передача решения в коллектор данных опыта
        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(root_state_tensor, visit_counts)

        #выбор хода с наибольшим количеством посещений
        return max(root.moves(), key=root.visit_count)

    def create_node(self, game_state, move=None, parent=None):
        """Создание нового узла в дереве поиска."""
        state_tensor = self.encoder.encode(game_state)
        #функция Keras predict - это функция пакетной обработки, принимающая
        #массив примеров. Поэтому мы должны обернуть board_tensor в массив.
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        #функция predict возращает массивы со множеством результатов, из которых
        #мы извлекаем первый элемент.
        priors = priors[0]
        # Добавить шум Дирихле к корневому узлу (root node) {гл.14.4}.
        if parent is None:
            noise = np.random.dirichlet(
                0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        value = values[0][0]
        #распаковка вектора априорных вероятностей в словарь, отображающий
        #объекты move в соответствующие априорные вероятности
        move_priors = {
            self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(
            game_state, value,
            move_priors,
            parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):
        """Обучение комбинированной сети.

        Params:
            learning_rate (lr) - скорость обучения
            batch_size - определяет кол.ходов из данных опыта, учитываемых при обновлении отдельного веса
        """

        num_examples = experience.states.shape[0]
        model_input = experience.states

        #нормализация значений счетчиков посещений. Вызов функции np.sum с использованием
        #axis=1 выполняет построчное суммирование элементов матрицы. Вызов функции reshape
        #реорганизует эти суммы в соответствующие строки. После этого мы можем разделить
        #исходные значения счетчиков посещений на их общее количество.
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sums

        value_target = experience.rewards

        self.model.compile(
            SGD(lr=learning_rate),
            loss=['categorical_crossentropy', 'mse'])

        self.model.fit(
            model_input, [action_target, value_target],
            batch_size=batch_size)
