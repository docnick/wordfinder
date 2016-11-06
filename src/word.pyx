
class Word:
    """
    Class encapsulates the path and the string of a word found on the board
    """

    def __init__(self, word="", path=[]):
        self._word_str = word
        if type(path) != list:
            path = [path]
        self._word_path = path
        self._len = len(word)

    def __add__(self, other):
        # we add to a tuple rather than another word
        word = self._word_str + other[0]
        path = self._word_path + [other[1]]
        self._len += 1
        return Word(word, path)

    def length(self):
        return self._len

    def get_path(self):
        return self._word_path

    def get_path_set(self):
        return set(self._word_path)

    def get_word(self):
        return self._word_str

    def __repr__(self):
        return self._word_str

    def __str__(self):
        return self._word_str
