from dlgo.goboard import Move


def is_ladder_capture(game_state, candidate, recursion_depth=50):
    return is_ladder(True, game_state, candidate, None, recursion_depth)


def is_ladder_escape(game_state, candidate, recursion_depth=50):
    return is_ladder(False, game_state, candidate, None, recursion_depth)


def is_ladder(try_capture, game_state, candidate,
              ladder_stones=None, recursion_depth=50):
    """Лестницы разыгрываются в поменявшихся ролях, один игрок пытается захватить,
    другой - сбежать. Мы определяем статус лестницы путем рекурсивного вызова is_ladder
    в противоположных ролях, предоставляя подходящих кандидатов для захвата или защиты.

    Args:
      try_capture: булеан флаг, указывающий, хотите ли вы захватить лестницу или сбежать из неё.
      game_state: текущее состояние игры, экземпляр GameState.
      candidate: ход, который потенциально приводит к выходу из лестницы или ее захвату, экземпляр Move.
      ladder_stones: камни для побега или захвата, список Point. Будет выведено, если не указано иное.
      recursion_depth: когда следует прекратить рекурсивный вызов этой функции, целочисленное значение.

    Returns: True, если игровое состояние является лестницей, а try_capture равно true (захват
    лестницы) или если игровое состояние не является лестницей, а try_capture равно false (вы
    можете успешно сбежать) и False в противном случае.
    """

    if not game_state.is_valid_move(Move(candidate)) or not recursion_depth:
        return False

    next_player = game_state.next_player
    capture_player = next_player if try_capture else next_player.other
    escape_player = capture_player.other

    if ladder_stones is None:
        ladder_stones = guess_ladder_stones(game_state, candidate, escape_player)

    for ladder_stone in ladder_stones:
        current_state = game_state.apply_move(candidate)

        if try_capture:
            candidates = determine_escape_candidates(
                game_state, ladder_stone, capture_player)
            attempted_escapes = [  # теперь попытайся сбежать
                is_ladder(False, current_state, escape_candidate,
                          ladder_stone, recursion_depth - 1)
                for escape_candidate in candidates]

            if not any(attempted_escapes):
                return True  # если хотя бы один побег не удастся, мы захватим
        else:
            if count_liberties(current_state, ladder_stone) >= 3:
                return True  # успешный побег
            if count_liberties(current_state, ladder_stone) == 1:
                continue  # неудачный побег, другие все еще могут это сделать
            candidates = liberties(current_state, ladder_stone)
            attempted_captures = [  # теперь попробуйте захватить
                is_ladder(True, current_state, capture_candidate,
                          ladder_stone, recursion_depth - 1)
                for capture_candidate in candidates]
            if any(attempted_captures):
                continue  # неудачный побег, попробуйте другие
            return True  # кандидат не может быть пойман на лестнице, сбежал.
    return False  # никаких захватов/побегов


def is_candidate(game_state, move, player):
    return game_state.next_player == player and \
        count_liberties(game_state, move) == 2


def guess_ladder_stones(game_state, move, escape_player):
    adjacent_strings = [game_state.board.get_go_string(nb) for nb in move.neighbors() if game_state.board.get_go_string(nb)]
    if adjacent_strings:
        string = adjacent_strings[0]
        neighbors = []
        for string in adjacent_strings:
            stones = string.stones
            for stone in stones:
                neighbors.append(stone)
        return [Move(nb) for nb in neighbors if is_candidate(game_state, Move(nb), escape_player)]
    else:
        return []


def determine_escape_candidates(game_state, move, capture_player):
    escape_candidates = move.neighbors()
    for other_ladder_stone in game_state.board.get_go_string(move).stones:
        for neighbor in other_ladder_stone.neighbors():
            right_color = game_state.color(neighbor) == capture_player
            one_liberty = count_liberties(game_state, neighbor) == 1
            if right_color and one_liberty:
                escape_candidates.append(liberties(game_state, neighbor))
    return escape_candidates


def count_liberties(game_state, move):
    if game_state.board.get_go_string(move):
        return game_state.board.get_go_string(move).num_liberties
    else:
        return 0


def liberties(game_state, move):
    return list(game_state.board.get_go_string(move).liberties)
