import math
import random

from dlgo import agent
from dlgo.gotypes import Player
from dlgo.utils import coords_from_point

def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    '''UCT - верхний предел доверительного интервала для деревьев.'''
    exporation = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exporation
    

class MCTSNode(object):
    '''Структура данных для представления дерева'''
    
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves()
    
    def add_random_child(self):
        '''Обновление узла дерева'''
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
    
    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1
    
    def can_add_child(self):
        '''Сообщает, предусматривает ли данная позиция допустимые ходы,
        которые еще не были добавлены в дерево.
        '''
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        '''Сообщает, заканчивается ли игра в данном узле.'''
        return self.game_state.is_over()
    
    def winning_frac(self, player): #он же winning_pct и victory_frac
        '''Возвращает процент выигрышных развертываний для данного игрока.'''
        return float(self.win_counts[player]) / float(self.num_rollouts)

class MCTSAgent(agent.Agent):
    '''Агент работающий по алгоритму Монте-Карло'''
    
    def __init__(self, num_rounds, temperature):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
    
    def select_move(self, game_state):
        '''Выбор лучшей ветви для исследования'''
        root = MCTSNode(game_state)
        
        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)
            
            if node.can_add_child():
                #добавление в дерево нового дочернего узла
                node = node.add_random_child()
            
            #развертывание случайной игры из этого узла
            winner = self.simulate_random_game(node.game_state)
            
            while node is not None:
                #обновление счета в предыдущих узлах дерева
                node.record_win(winner)
                node = node.parent
    
        #выбираем ход после развертываний, выбирается который имеет наибольший процент выигрышей
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node):
        '''Выбор ветви для исследования с помощью формулы UCT'''
        total_rollouts = sum(child.num_rollouts for child in node.children)
        
        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_frac(node.game_state.next_player),
                self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        """Симуляция игры как 'bot vs bot'"""
        bots = {
            Player.black: agent.RandomBot(), #.FastRandomBot(),
            Player.white: agent.RandomBot(), #.FastRandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()