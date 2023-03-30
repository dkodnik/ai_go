import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move
from dlgo.gotypes import Player, Point


class SimpleEncoder(Encoder):
    def __init__(self, board_size):
        """
        Args:
            board_size: кортеж из (ширина, высота)
        """
        self.board_width, self.board_height = board_size
        # 0 - 3. черные камни с 1, 2, 3, 4+ степенями свободы (liberty)
        # 4 - 7. белые камни с 1, 2, 3, 4+ степенями свободы (liberty)
        # 8. следующими ходят черные
        # 9. следующими ходят белые
        # 10. зарезервировано для Ко
        self.num_planes = 11

    def name(self):
        return 'simple'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        if game_state.next_player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_point(self, point):
        """Преобразует точку на доске в целочисленный индекс."""
        # Точки проиндексированы на 1
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Преобразует целочисленный индекс в точку на доске."""
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return SimpleEncoder(board_size)
