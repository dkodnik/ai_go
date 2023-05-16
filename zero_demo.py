from dlgo.goboard_fast import GameState, Player
from dlgo import scoring

#from dlgo.goboard_fast import Point
#from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from dlgo import zero


def simulate_game(board_size, black_agent, black_collector, white_agent, white_collector):
    """Симуляция игры бота с самим собой."""
    print('Старт игры!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }
    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)

## Один цикл обучения с подкреплением
# Для достижения сверхчеловеческого уровня игры системе AlphaGo Zero
# потребовалось сыграть около 5 млн. игр с самой собой.

board_size = 9
encoder = zero.ZeroEncoder(board_size)

board_input = Input(shape=encoder.shape(), name='board_input')
pb = board_input
#создайте сеть с 4-мя сверточными слоями. Для создания сильного бота можно
#добавить множество дополнительных слоев
for i in range(4):
    pb = Conv2D(64, (3,3),
                padding='same',
                data_format='channels_first',
                activation='relu')(pb)

#добавьте в сеть выход функции политики
policy_conv = Conv2D(2, (1,1),
                     data_format='channels_first',
                     activation='relu')(pb)
policy_flat = Flatten()(policy_conv)
policy_output = Dense(encoder.num_moves(), activation='softmax')(policy_flat)

#добавьте в сеть выход функции ценности
value_conv = Conv2D(1, (1,1),
                    data_format='channels_first',
                    activation='relu')(pb)
value_flat = Flatten()(value_conv)
value_hidden = Dense(256, activation='relu')(value_flat)
value_output = Dense(1, activation='tanh')(value_hidden)

model = Model(
    inputs=[board_input],
    outputs=[policy_output, value_output])

#здесь мы используем 10 раундов на ход для ускорения работы демонстрационной
#версии. В ходе реального обучения потребляется гораздо большее их количество.
#Система AGZ использовала 1600.
black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
c1 = zero.ZeroExperienceCollector()
c2 = zero.ZeroExperienceCollector()
black_agent.set_collector(c1)
white_agent.set_collector(c2)

#сгенерируйте 5-ть игр перед началом обучения. В ходе реального обучения
#вам потребуются партии, состоящие из тысяч игр.
for i in range(5):
    simulate_game(board_size, black_agent, c1, white_agent, c2)

exp = zero.combine_experience([c1, c2])
black_agent.train(exp, 0.01, 2048)