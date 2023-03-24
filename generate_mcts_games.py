'''Пример:
Допустим, нам нужно сгенерировать данные для двадцати игр на доске 9×9, со-
хранить признаки в файле features.npy, а метки – в файле label.npy. Для этого мож-
но использовать следующую команду:

python generate_mcts_games.py -n 20 --board-out features.npy --move-out labels.npy
'''

import argparse
import numpy as np

from dlgo.encoders import get_encoder_by_name
#from dlgo import goboard_fast as goboard
from dlgo import goboard
from dlgo.mcts import mcts
from dlgo.utils import print_board, print_move

def generate_game(board_size, rounds, max_moves, temperature):
    '''Генератор игровых данных методом Монте-Карло'''
    
    #boards - закодированное состояние доски
    #moves - закодированные ходы
    boards, moves = [], []
    
    #инициализация кодировщика OnePlaneEncoder и размер доски
    encoder = get_encoder_by_name('oneplane', board_size)
    
    #создание новой игры
    game = goboard.GameState.new_game(board_size)
    
    #в качестве бота будет агент поиска по дереву методом Монте-Карло
    #с указанием кол.раундов и температуры
    bot = mcts.MCTSAgent(rounds, temperature)
    
    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game) #след.ход выбирается ботом
        if move.is_play:
            #закодированные данные о состоянии доски добавляются в boards
            boards.append(encoder.encode(game))
            
            #след.ход закодированный методом унитарного кодирования...
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot) #...добавляются в moves
        
        print_move(game.next_player, move)
        game = game.apply_move(move) #ход бота применяется к доске
        num_moves += 1
        if num_moves > max_moves:
            #цикл повторяется, пока не достигнуто max кол.ходов
            break
    
    return np.array(boards), np.array(moves)

def main():
    #-n 20 --board-out features.npy --move-out labels.npy
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=20) #def=10
    parser.add_argument('--board-out', type=str, default='features.npy') #без type и def
    parser.add_argument('--move-out', type=str, default='labels.npy') #без type и def

    args = parser.parse_args()
    xs = []
    ys = []
    
    for i in range(args.num_games):
        print('Generating game %d/%d...' % (i+1, args.num_games))
        #генерирование игровых данных
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)
    
    #после генерации всех игр производится конкатенация признаков и меток
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    
    #данные о признаках и метках сохраняем в отдельные файлы
    np.save(args.board_out, x)
    np.save(args.move_out, y)

if __name__ == '__main__':
    main()
    