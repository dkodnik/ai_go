import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Point

class OnePlaneEncoder(Encoder):
    '''Одноплоскостной кодировщик доски'''
    
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1
    
    def name(self):
        #Ссылаемся на этот кодирощик, указав имя oneplane
        return 'oneplane'
    
    def encode(self, game_state):
        '''Кодирование. Помещаем в матрицу: 
        1-если в точке доски находится камень следующего игрока (кто именно будет ходить - текущего но еще не походившего!)
        -1-если в точке доски находится камень противника
        0-если точка доски пуста
        '''
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r+1, col=c+1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix
    
    def encode_point(self, point):
        #Преобразуем точку доски в целочисленный индекс
        return self.board_width * (point.row - 1) + (point.col - 1)
    
    def decode_point_index(self, index):
        #Преобразуем целочисленный индекс в точку доски
        row = index // self.board_width
        col  = index % self.board_width
        return Point(row=row+1, col=col+1)
    
    def num_points(self):
        return self.board_width * self.board_height
    
    def shape(self):
        return self.num_planes, self.board_height, self.board_width