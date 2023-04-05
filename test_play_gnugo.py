"""Тест игры бота с программой GNU Go или Pachi.
"""
from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import h5py

bot = load_prediction_agent(h5py.File("./agents/sevenplane.h5", "r"))
gnu_go = LocalGtpBot(go_bot=bot, termination=PassWhenOpponentPasses(),
                     handicap=0, opponent='gnugo', ) #opponent='pachi'
gnu_go.run()