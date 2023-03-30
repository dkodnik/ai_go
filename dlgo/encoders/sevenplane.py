import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move, Point

class SevenPlaneEncoder(Encoder):
    """Простой кодировщик состоящий из 7 плоскостей.
    0пл - для белых у которых одна степенью свободы, 1-есть, 0-нету
    1пл, 2пл - для белых с двумя или как минимум тремя степенями свободы, 1-есть, 0-нету
    3пл - для черных у которых одна степенью свободы, 1-есть, 0-нету
    4пл, 5пл - для черных с двумя или как минимум тремя степенями свободы, 1-есть, 0-нету
    6пл - кодирует значением 1 точки доски, помещать камни на которые - запрещено правилом Ко.
    """

    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self):
        return 'sevenplane'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        base_plane = {game_state.next_player: 0,
                      game_state.next_player.other: 3}
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row+1, col=col+1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        #кодирование ходов, запрещенных правилом Ко
                        board_tensor[6][row][col] = 1
                    else:
                        liberty_plane = min(3, go_string.num_liberties) - 1
                        liberty_plane += base_plane[go_string.color]
                        #кодирование черных и белых камней с 1,2 или большим количеством степеней свободы
                        board_tensor[liberty_plane][row][col] = 1
        return board_tensor

    def encode_point(self, point):
        return self.board_width * (point.row-1) + (point.col-1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

def create(board_size):
    return SevenPlaneEncoder(board_size)