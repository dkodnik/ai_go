"""Идентичен кодировщику - 'SevenPlaneEncoder'.
"""

import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move, Point


class BetaGoEncoder(Encoder):
    """7 плоскостной энкодер, используемый в betago"""
    def __init__(self, board_size):
        """
        Args:
            board_size: кортеж из ширина, высота (width, height)
        """
        self.board_width, self.board_height = board_size
        # 0 - 2. наш камень с  1, 2, 3+ степенями свободы
        # 3 - 5. камень противника с 1, 2, 3+ степенями свободы
        # 6. незаконные ходы из-за Ко.
        self.num_planes = 7

    def name(self):
        return 'betago'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        base_plane = {
            game_state.next_player: 0,
            game_state.next_player.other: 3,
        }
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        # кодирование ходов, запрещенных правилом Ко
                        board_tensor[6][r][c] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    # кодирование черных и белых камней с 1,2 или большим количеством степеней свободы
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_point(self, point):
        """Преобразует точку на доске в целочисленный индекс."""
        # Points are 1-indexed
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
    return BetaGoEncoder(board_size)
