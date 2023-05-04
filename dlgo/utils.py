import numpy as np
from dlgo import gotypes

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ',
    gotypes.Player.white: ' o ',
}

def print_move(player, move):
    """Выводит следующий ход"""
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))

def print_board(board):
    """Выводит текущее состояние доски со всеми расположенными на ней камнями"""
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))

def point_from_coords(coords):
    """Преобразование строки (например: C3 или E7) в координаты доски"""
    col = COLS.index(coords[0].upper()) + 1 #вдруг ввели в нижнем - переведем символ в верхний регистр
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)

def coords_from_point(point):
    """Преобразование координаты в строку"""
    return '%s%d' % (COLS[point.col - 1], point.row)

# ПРИМЕЧАНИЕ: MoveAge используется только в главе 13 и не попадает в основной текст.
# Эта функция будет реализована только в 'goboard_fast.py' (goboard.py) чтобы не вводить
# читателей в заблуждение в первых главах.
class MoveAge():
    def __init__(self, board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1