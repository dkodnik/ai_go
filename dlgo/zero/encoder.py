import numpy as np
from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Move
from dlgo.gotypes import Point, Player

class ZeroEncoder(Encoder):
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3. наши камни с 1, 2, 3, 4+ степенями свободы (liberty)
        # 4 - 7. камни противника с 1, 2, 3, 4+ степенями свободы (liberty)
        # 8. 1 если мы получим Коми
        # 9. 1, если противник получит Коми
        # 10. движение было бы незаконно из-за Ко.
        self.num_planes = 11

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        if game_state.next_player == Player.white:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color != next_player:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_move(self, move):
        """Кодировщик доски с учетом пропуска хода."""
        if move.is_play:
            #точки доски кодируются так же, как в предыдущих кодировщиках
            return (self.board_size * (move.point.row - 1) + (move.point.col - 1))
        elif move.is_pass:
            #пропуск хода кодируется с помощью индекса следующего элемента
            return self.board_size * self.board_size
        #сообщение, горорящее о том, что нейронная сеть не научилась выходить из игры
        raise ValueError('Cannot encode resign move')

    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row+1, col=col+1))

    def num_moves(self):
        return self.board_size * self.board_size + 1

    def shape(self):
        return self.num_planes, self.board_size, self.board_size