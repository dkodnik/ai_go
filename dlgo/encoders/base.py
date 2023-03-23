import importlib
class Encoder:
    def name(self):
        # Название кодировщика
        raise NotImplementedError()
    
    def encode(self, game_state):
        # Состояние доски в числовые данные
        raise NotImplementedError()
    
    def encode_point(self, point):
        # Точка доски в целочисленный индекс
        raise NotImplementedError()
    
    def decode_point_index(self, index):
        # Целочисленный индекс в точку доски
        raise NotImplementedError()
    
    def num_points(self):
        # Количество точек на доске = ширина доски * высоту доски
        raise NotImplementedError()
    
    def shape(self):
        # Форма закодированной структуры доски
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):
    # Экземпляры кодировщика можно создавать путем указания из имени
    if isinstance(board_size, board_size):
        board_size = (board_size, board_size)
    module = importlib.import_module('dlgo.encoders.'+name)
    constructor = getattr(module, 'create')
    return constructor(board_size)