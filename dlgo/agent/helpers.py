from dlgo.gotypes import Point

def is_point_an_eye(board, point, color):
    '''Является ли "глазом" данная точка доски?'''
    if board.get(point) is not None:
        #Глаз - это пустая точка, поэтому это не глаз!
        return False
    #Все смежные точки должны содержать дружественные камни
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    
    #Мы должны контролировать 3 из 4 углов, если точка находится в середине доски.
    #Если точка находится на краю доски, то мы должны контролировать все углы.
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        #Точка находится на краю или в углу доски:
        return off_board_corners + friendly_corners == 4
    #Точка находится в середине доски:
    return friendly_corners >= 3
