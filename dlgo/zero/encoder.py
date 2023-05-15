from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Move
from dlgo.gotypes import Point

class ZeroEncoder(Encoder):
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. opponent stones with 1, 2, 3, 4+ liberties
        # 8. 1 if we get komi
        # 9. 1 if opponent gets komi
        # 10. move would be illegal due to ko
        self.num_planes = 11

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

    def decode_move(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row+1, col=col+1))

    def num_moves(self):
        return self.board_size * self.board_size + 1

    def shape(self):
        return self.num_planes, self.board_size, self.board_size