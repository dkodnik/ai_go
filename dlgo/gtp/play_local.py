"""Запуск игры бота против других ботов"""

import subprocess
import re
#import h5py

#from dlgo.agent.predict import load_prediction_agent
from dlgo.agent.termination import PassWhenOpponentPasses, TerminationAgent
from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player
from dlgo.gtp.board import gtp_position_to_coords, coords_to_gtp_position
from dlgo.gtp.utils import SGFWriter
from dlgo.utils import print_board
from dlgo.scoring import compute_game_result


class LocalGtpBot:
    """Утилита для запуска игры двух ботов."""

    def __init__(self, go_bot, termination=None, handicap=0,
                 opponent='gnugo', output_sgf='out.sgf',
                 our_color='b'):
        #инициализация бота с использованием агента и стратегии прекращения игры
        self.bot = TerminationAgent(go_bot, termination)
        self.handicap = handicap
        #игра продолжается до остановки одного из игроков
        self._stopped = False
        self.game_state = GameState.new_game(19)
        #в конце партии записывается в файл в формате SGF
        self.sgf = SGFWriter(output_sgf)

        self.our_color = Player.black if our_color == 'b' else Player.white
        self.their_color = self.our_color.other

        #в качестве противника нашего бота может использоваться GNU Go или Pachi
        cmd = self.opponent_cmd(opponent)
        pipe = subprocess.PIPE
        #чтение и запись GTP-команд из командной строки
        self.gtp_stream = subprocess.Popen(cmd, stdin=pipe, stdout=pipe)

    @staticmethod
    def opponent_cmd(opponent):
        if opponent == 'gnugo':
            return ["gnugo", "--mode", "gtp"]
        elif opponent == "pachi":
            return ["pachi"]
        else:
            raise ValueError("Unknown bot name {}".format(opponent))

    def send_command(self, cmd):
        """Отправка GTP-команды"""
        self.gtp_stream.stdin.write(cmd.encode('utf-8'))

    def get_response(self):
        """Получение ответа от внешнего бота по GTP"""
        succeeded = False
        result = ''
        while not succeeded:
            line = self.gtp_stream.stdout.readline()
            if line[0] == '=':
                succeeded = True
                line = line.strip()
                result = re.sub('^= ?', '', line)
        return result

    def command_and_response(self, cmd):
        """Отправка GTP-команды и получение ответа"""
        self.send_command(cmd)
        return self.get_response()

    def run(self):
        """Настройка доски, игра и сохранение записи партии"""

        #настройка доски, создание её размером 19x19
        self.command_and_response("boardsize 19\n")
        #задается гандикап
        self.set_handicap()
        #запуск игры (игровой логики)
        self.play()
        #запись партии сохраняется в файле SGF
        self.sgf.write_sgf()

    def set_handicap(self):
        """Задание Гандикапа"""
        if self.handicap == 0:
            self.command_and_response("komi 7.5\n")
            self.sgf.append("KM[7.5]\n")
        else:
            stones = self.command_and_response("fixed_handicap {}\n".format(self.handicap))
            sgf_handicap = "HA[{}]AB".format(self.handicap)
            for pos in stones.split(" "):
                move = gtp_position_to_coords(pos)
                self.game_state = self.game_state.apply_move(move)
                sgf_handicap = sgf_handicap + "[" + self.sgf.coordinates(move) + "]"
            self.sgf.append(sgf_handicap + "\n")

    def play(self):
        """Игра.
        Игра заканчивается, когда один из противников подает соответствующий сигнал.
        """
        while not self._stopped:
            if self.game_state.next_player == self.our_color:
                self.play_our_move()
            else:
                self.play_their_move()
            print(chr(27) + "[2J")
            print_board(self.game_state.board)
            print("Estimated result: ")
            print(compute_game_result(self.game_state))

    def play_our_move(self):
        """Команда заставляет бота """
        move = self.bot.select_move(self.game_state)
        self.game_state = self.game_state.apply_move(move)

        our_name = self.our_color.name
        our_letter = our_name[0].upper()
        sgf_move = ""
        if move.is_pass:
            self.command_and_response("play {} pass\n".format(our_name))
        elif move.is_resign:
            self.command_and_response("play {} resign\n".format(our_name))
        else:
            pos = coords_to_gtp_position(move)
            self.command_and_response("play {} {}\n".format(our_name, pos))
            sgf_move = self.sgf.coordinates(move)
        self.sgf.append(";{}[{}]\n".format(our_letter, sgf_move))

    def play_their_move(self):
        """Противник совершает ходы в ответ на команду genmove"""
        their_name = self.their_color.name
        their_letter = their_name[0].upper()

        pos = self.command_and_response("genmove {}\n".format(their_name))
        if pos.lower() == 'resign':
            self.game_state = self.game_state.apply_move(Move.resign())
            self._stopped = True
        elif pos.lower() == 'pass':
            self.game_state = self.game_state.apply_move(Move.pass_turn())
            self.sgf.append(";{}[]\n".format(their_letter))
            if self.game_state.last_move.is_pass:
                self._stopped = True
        else:
            move = gtp_position_to_coords(pos)
            self.game_state = self.game_state.apply_move(move)
            self.sgf.append(";{}[{}]\n".format(their_letter, self.sgf.coordinates(move)))