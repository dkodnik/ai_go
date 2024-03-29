import copy
from dlgo.gotypes import Player

class Move():
    '''Ход
    '''
    
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign
    
    @classmethod
    def play(cls, point):
        '''Ход предполагает размещение камня на доске'''
        return Move(point=point)
    
    @classmethod
    def pass_turn(cls):
        '''Ход предполагает пропуск хода'''
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):
        '''Ход предполагает выход из игры'''
        return Move(is_resign=True)


class GoString():
    '''Цепочка связанных камней одного цвета
    '''
    
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)
    
    def remove_liberty(self, point):
        '''Удалить степень свободы цепочки'''
        self.liberties.remove(point)
    
    def add_liberty(self, point):
        '''Добавить степень свободы к цепочке'''
        self.liberties.add(point)
    
    def merged_with(self, go_string):
        '''Возвращает новую цепочку, содержащую все камни обеих цепочек'''
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones)
    
    @property
    def num_liberties(self):
        return len(self.liberties)
    
    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties


class Board():
    '''Доска
    '''
    
    def __init__(self, num_rows, num_cols):
        '''Инициализация в виде пустой сетки, состоящая из заданного количества строк и столбцов'''
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
    
    def place_stone(self, player, point):
        '''Проверка количества степеней свободы соседних точек'''
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
        for neighbor in point.neighbors():
            #сначала исследуются непосредственные соседи конкретной точки
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)
        
        #Объединение всех смежных цепочек камней одного цвета
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        #Уменьшение количества степеней свободы соседних цепочек камней противоположного цвета
        for other_color_string in adjacent_opposite_color:
            other_color_string.remove_liberty(point)
        #Удаление с доски цепочек камней противоположного цвета с нулевой степенью свободы
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)
    
    def is_on_grid(self, point):
        '''Проверка на размещение на доске'''
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols
    
    def get(self, point):
        '''Возвращает содержимое точки на доске: сведения об игроке, если в этой точке находится камень, в противном случае - None'''
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color
    
    def get_go_string(self, point):
        '''Возвращает всю цепочку камней, если в этой точке находится камень, в противном случае - None'''
        string = self._grid.get(point)
        if string is None:
            return None
        return string
    
    def _remove_string(self, string):
        '''Удаление камней'''
        for point in string.stones:
            #Удаление цепочки может привести к увеличению степеней свободы других цепочек
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    neighbor_string.add_liberty(point)
            self._grid[point] = None


class GameState():
    '''Игровое состояние
    '''
    
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
    
    def apply_move(self, move):
        '''Возвращает новое игровое состояние после совершения хода'''
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)
    
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
    
    def is_over(self):
        '''Проверка окончания игры'''
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass
    
    def is_move_self_capture(self, player, move):
        '''Проверка самозахвата'''
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board) #проверка будет на копие Доски
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0
    
    @property
    def situation(self):
        return (self.next_player, self.board)
    
    def does_move_violate_ko(self, player, move):
        '''Проверка на нарушение состояния Ко'''
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        past_state = self.previous_state
        while past_state is not None:
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False
    
    def is_valid_move(self, move):
        '''Проверка на допустимость хода для данного игрового состояния'''
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move))
