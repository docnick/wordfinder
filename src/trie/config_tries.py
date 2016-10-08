from functools import lru_cache
import datrie
import string
import os

TRIE_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../word_data/')
WORD_FILE = os.path.join(TRIE_FILE_PATH, 'words.dat')
TRIE_FILE_PREFIX = 'en_words'
TRIE_EXT = 'trie'
TRIE = datrie.Trie(string.ascii_lowercase)
MAX_WORD_LEN = 9


def build_trie_path(word_len):
    # if word_len >= MAX_WORD_LEN:
    #     word_len = '9+'
    full_path = os.path.join(TRIE_FILE_PATH, '{}-{}.{}'.format(TRIE_FILE_PREFIX, word_len, TRIE_EXT))
    return full_path


def _get_word_len_from_file(file_name):
    idx1 = file_name.find('-')
    idx2 = file_name.find('.{}'.format(TRIE_EXT))

    num = file_name[idx1+1:idx2]
    return num


def load_tries(files):
    tries = {}
    for trie_file in files:
        trie = datrie.Trie.load(trie_file)
        num = _get_word_len_from_file(trie_file)
        print('loading {} for trie[{}]'.format(trie_file, num))
        tries[num] = trie

    return tries


@lru_cache(16)
def _equals(a, b):
    return a == b


@lru_cache(16)
def _gte(a, b):
    return a >= b


def _build_trie(word_len=None, compare_func=_gte):
    trie = datrie.Trie(string.ascii_lowercase)

    with open(WORD_FILE, 'r') as fin:
        for line in fin:
            line = line.strip()
            if word_len is None or compare_func(len(line), word_len):
                trie[u'{}'.format(line)] = 1
    return trie


def _build_tries(word_lens):

    for word_len in word_lens:
        trie = _build_trie(word_len)
        print('saving trie[{}] with {} words'.format(word_len, len(trie)))
        full_path = build_trie_path(word_len)
        trie.save(full_path)


if __name__ == '__main__':
    print('build tries...')
    trie_sizes = list(range(2, MAX_WORD_LEN + 1))
    _build_tries(trie_sizes)

    # trie = _build_trie(MAX_WORD_LEN, _gte)
    # full_path = build_trie_path(MAX_WORD_LEN)
    # trie.save(full_path)
