"""Реализация GPT-команды."""

class Command:

    def __init__(self, sequence, name, args):
        self.sequence = sequence
        self.name = name
        self.args = tuple(args)

    def __eq__(self, other):
        return self.sequence == other.sequence and \
            self.name == other.name and \
            self.args == other.args

    def __repr__(self):
        return 'Command(%r, %r, %r)' % (self.sequence, self.name, self.args)

    def __str__(self):
        return repr(self)

def parse(command_string):
    """Parse a GTP protocol line into a Command object.
    Преобразование простого текста в GTP-команду

    Example:
    >>> parse('999 play white D4')
    Command(999, 'play', ('white', 'D4'))
    """

    pieces = command_string.split()
    try:
        #GTP-команды могут начинаться с порядкового номера
        sequence = int(pieces[0])
        pieces = pieces[1:]
    except ValueError:
        sequence = None
    name, args = pieces[0], pieces[1:]
    return Command(sequence, name, args)