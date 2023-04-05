"""Преобразование координат GTP в данные типа Point."""

from dlgo.gotypes import Point
from dlgo.goboard import Move

__all__ = [
    'coords_to_gtp_position',
    'gtp_position_to_coords',
]

COLS = 'ABCDEFGHJKLMNOPQRST'

def coords_to_gtp_position(move):
    """Convert (row, col) tuple to GTP board locations.

    Example:
    >>> coords_to_gtp_position((1, 1))
    'A1'
    """

    point = move.point
    return COLS[point.col-1] + str(point.row)

def gtp_position_to_coords(gtp_position):
    """Convert a GTP board location to a (row, col) tuple.

    Example:
    >>> gtp_position_to_coords('A1')
    (1, 1)
    """

    col_str, row_str = gtp_position[0], gtp_position[1:]
    point = Point(int(row_str), COLS.find(col_str.upper())+1)
    return Move(point)