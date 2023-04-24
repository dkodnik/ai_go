import h5py

from dlgo import goboard
from dlgo import gotypes
from dlgo import rl
from dlgo import scoring
from dlgo.agent.pg import load_policy_agent

BOARD_SIZE = 9
num_games = 10
experience_filename = "ex.h5"
agent_filename = "./agents/sevenplane.h5"

def simulate_game(black_player, white_player):
    game = goboard.GameState.new_game(BOARD_SIZE)
    agents = {
        gotypes.Player.black: black_player,
        gotypes.Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    game_result = scoring.compute_game_result(game)
    return game_result.winner

def main():
    agent1 = load_policy_agent(h5py.File(agent_filename))
    agent2 = load_policy_agent(h5py.File(agent_filename))
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == gotypes.Player.black:
            #агент agent1 победил, поэтому он получает положительную награду
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            # агент agent2 победил
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)

        #сохранение пакета данных опыта, со объединением данных опыта обоих агентов в единый буфер
        experience = rl.load_experience([
            collector1,
            collector2])
        with h5py.File(experience_filename, 'w') as experience_outf: #сохранение результата в файд HDF5
            experience.serialize(experience_outf)


if __name__ == '__main__':
    main()