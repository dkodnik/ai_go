import h5py

from dlgo.agent.pg import load_policy_agent
from dlgo.rl.simulate import experience_simulation

num_games = 1000
experience_filename = "alphago_rl_experience.h5"
agent_filename = "./agents/sevenplane.h5"

def main():
    agent1 = load_policy_agent(h5py.File(agent_filename))
    agent2 = load_policy_agent(h5py.File(agent_filename))

    experience = experience_simulation(num_games, agent1, agent2)

    with h5py.File(experience_filename, 'w') as experience_outf: #сохранение результата в файд HDF5
        experience.serialize(experience_outf)

if __name__ == '__main__':
    main()