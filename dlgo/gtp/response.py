"""Кодирование и сериализация GTP-ответа"""

class Response:
    def __init__(self, status, body):
        self.success = status
        self.body = body

def success(body=''):
    """Создание GTP-ответа с телом ответа на случай
    удачного выполнения команды.
    """
    return Response(status=True, body=body)

def error(body=''):
    """Создание GTP-ответа на случай неудачного
    выполнения команды.
    """
    return Response(status=False, body=body)

def bool_response(boolean):
    """Преобразование логического значения
    Python в GTP.
    """
    return success('true') if boolean is True else success('false')

def serialize(gtp_command, gtp_response):
    """Сериализация GTP-ответа в строку."""
    return '{}{} {}\n\n'.format(
        '=' if gtp_response.success else '?',
        '' if gtp_command.sequence is None else str(gtp_command.sequence),
        gtp_response.body
    )