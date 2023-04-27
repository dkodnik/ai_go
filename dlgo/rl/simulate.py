from dlgo import goboard
from dlgo import gotypes
from dlgo import rl
from dlgo import scoring

#from collections import namedtuple
#class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
#    pass

def simulate_game(black_player, white_player):
    board_size = 19
    #moves = []
    game = goboard.GameState.new_game(board_size)
    agents = {
        gotypes.Player.black: black_player,
        gotypes.Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
        #moves.append(next_move)
    game_result = scoring.compute_game_result(game)
    print("Победил:", game_result.winner)
    return game_result
    #return GameRecord(
    #    moves=moves,
    #    winner=game_result.winner,
    #    margin=game_result.winning_margin,
    #)

def experience_simulation(num_games, agent1, agent2):
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        print(f'Игра {i+1}/{num_games}')
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == gotypes.Player.black:
            # агент agent1 победил, поэтому он получает положительную награду
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            # агент agent2 победил
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)

    # сохранение пакета данных опыта, с объединением данных опыта обоих агентов в единый буфер
    experience = rl.combine_experience([
        collector1,
        collector2])
    return experience