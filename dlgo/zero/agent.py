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