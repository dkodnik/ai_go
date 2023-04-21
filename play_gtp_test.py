"""Тест GTP-интерфейса с запуском из командной строки
"""

import h5py

from dlgo.gtp import frontend
from dlgo.agent.predict import load_prediction_agent
from dlgo.agent import termination

model_file = h5py.File("./agents/sevenplane.h5", "r")
agent = load_prediction_agent(model_file)
strategy = termination.get("opponent_passes")
termination_agent = termination.TerminationAgent(agent, strategy)

front = frontend.GTPFrontend(termination_agent)
front.run()