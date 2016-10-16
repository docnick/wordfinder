import numpy as np
from collections import defaultdict
import time
import threading
import image_utils
import trie.config_tries as trie
import file_utils as futil
from functools import lru_cache


MOVES = np.array([-1, 0, 1])
EMPTY_CELL = '@'
TRIES = {}

# file where we look for user hints
HINT_FILE = 'hints.txt'
# if hint is prefixed with an 'x', it should be excluded,
# if it is prefixed with an 'i', it should be included
# e.g. x:cupboards -> cupboards is NOT part of the solution
EXCLUDE_PREFIX = 'x'
INCLUDE_PREFIX = 'i'

EXCLUDE_SET = set()
SOLUTION_SET = set()
SOLUTION_DICT = {}
IS_NEW_HINT = False


class Word:
    """
    Class encapsulates the path and the string of a word found on the board
    """

    def __init__(self, word="", path=[]):
        self._word_str = word
        if type(path) != list:
            path = [path]
        self._word_path = path

    def __add__(self, other):
        # we add to a tuple rather than another word
        word = self._word_str + other[0]
        path = self._word_path + [other[1]]
        return Word(word, path)

    def length(self):
        return len(self._word_str)

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


@lru_cache(32)
def _transition(position, height, width):

    valid_moves = []
    for x in position[1] + MOVES:
        if x < 0 or x >= width:
            continue

        for y in position[0] + MOVES:

            if y < 0 or y >= height:
                continue

            # skip the current position
            if x == position[1] and y == position[0]:
                continue

            valid_moves.append((y, x))

    return valid_moves


@lru_cache(16)
def _get_trie(word_len):
    if word_len >= trie.MAX_WORD_LEN:
        word_len = trie.MAX_WORD_LEN

    return TRIES.get(str(word_len))


def find_paths(matrix, position, path_length, solution_word=None, exclude_set=set()):
    """
    Iterative algorithm for finding all valid words of a given length given a starting position
    :param matrix:
    :param position:
    :param path_length:
    :param exclude_set:
    :return:
    """
    height, width = matrix.shape

    paths = []
    stack = [Word(matrix[position[0], position[1]], position)]
    while stack:

        word = stack.pop()
        pos_set = word.get_path_set()
        for neighbor in _transition(word.get_path()[-1], height, width):
            if neighbor in pos_set:
                # can't revisit a letter
                continue

            v = matrix[neighbor[0], neighbor[1]]
            if v == EMPTY_CELL:
                continue

            tmp_word = word + (v, neighbor)
            trie = _get_trie(tmp_word.length())

            # we know the solution word!
            if solution_word:
                if tmp_word.get_word() != solution_word[:tmp_word.length()]:
                    continue

            if trie is None:
                # something bad happened
                print(tmp_word)

            if trie.has_keys_with_prefix(str(tmp_word)):
                if tmp_word.length() == path_length:
                    if str(tmp_word) not in exclude_set and str(tmp_word) in trie:
                        paths.append(tmp_word)

                elif tmp_word.length() < path_length:
                    stack.append(tmp_word)

    return paths


def evaluate_all_words(matrix, word_length, solution_word=None, exclude_set=set()):
    height, width = matrix.shape

    for i in range(width):
        for j in range(height):
            if matrix[j, i] == EMPTY_CELL:
                continue

            if solution_word:
                if matrix[j, i] != solution_word[0]:
                    continue

            start = (j, i)
            for word in find_paths(matrix, start, word_length, solution_word, exclude_set):
                yield word


def remove_word(board, word):
    height, width = board.shape

    # remove letters
    colums_to_fix = set()
    for letter_pos in word:
        board[letter_pos[0], letter_pos[1]] = EMPTY_CELL
        colums_to_fix.add(letter_pos[1])

    for i in colums_to_fix:
        for j in range(height - 1, -1, -1):
            if board[j, i] != EMPTY_CELL:
                board = _drop_letter(board, (j, i))

    return board


def _drop_letter(board, pos):
    height, width = board.shape

    letter = board[pos[0], pos[1]]
    board[pos[0], pos[1]] = EMPTY_CELL

    found_empty = False
    for j in range(pos[0], height):
        if board[j, pos[1]] != EMPTY_CELL:
            found_empty = True
            break

    if found_empty:
        board[j - 1, pos[1]] = letter
    else:
        board[j, pos[1]] = letter
    return board


def _is_solution_valid(solution, exclude_set, solution_set=set()):
    ss = set([str(s) for s in solution])

    if len(ss - exclude_set) == len(ss):
        if solution_set and len(ss - solution_set) == len(ss) - len(solution_set) or not solution_set:
            return True
    return False


def _valid_letters(solution_letters, letter_counts):

    for letter, count in solution_letters.items():
        letter_counts[letter] -= count
        if letter_counts[letter] < 0:
            return False

    return True


def _get_diff(solution, solution_set):
    letters = defaultdict(int)

    for word in solution:
        if word.get_word() not in solution_set:
            for w in list(str(word)):
                letters[w] += 1

    return letters


def _is_solution_valid_with_hints(solution, solution_dict):

    for i, word in enumerate(solution):
        if solution_dict.get(i) is not None:
            if word.get_word() != solution_dict.get(i):
                return False

    return True


def _purge_stack(stack, letter_counts, exclude_set, solution_set):
    purged_stack = []

    board_letter_count = letter_counts.copy()
    for word in solution_set:
        for w in list(word):
            board_letter_count[w] -= 1

    for board, word_lengths, solution, solution_word in stack:
        if _is_solution_valid(solution, exclude_set):

            if _is_solution_valid_with_hints(solution, SOLUTION_DICT):
                solution_offset = len(solution)
                solution_word = SOLUTION_DICT.get(solution_offset)
                purged_stack.append((board, word_lengths, solution, solution_word))

            #
            # Since we don't know the order of the words in the solution set, we just remove the ability
            # of using those letters in part of another word. For example if our board contains 2 'z's
            # and we know that 'puzzle' is part of the solution set, we shouldn't accept any other words
            # containing a 'z' (e.g. 'zoo') in a solution since these letters are already used up
            # if _valid_letters(_get_diff(solution, solution_set), board_letter_count.copy()):
            #
            #     purged_stack.append((board, word_lengths, solution))

    return purged_stack


def _purge_results(results, letter_counts, exclude_set, solution_set):
    purged_results = []

    board_letter_count = letter_counts.copy()
    for word in solution_set:
        for w in list(word):
            board_letter_count[w] -= 1

    for solution in results:
        if _is_solution_valid(solution, exclude_set):

            if _is_solution_valid_with_hints(solution, SOLUTION_DICT):
                purged_results.append(solution)

    return purged_results


def _get_letter_count(board):
    letters = defaultdict(int)
    for l in board.flatten().tolist()[0]:
        letters[l] += 1
    return letters


def solve_board(board, word_lengths, solution=[]):
    global IS_NEW_HINT
    iters = 0
    results = []

    s = time.time()
    letter_count = _get_letter_count(board)
    stack = [(board, word_lengths, solution, None)]
    try:
        while stack:
            board, word_lengths, solution, solution_word = stack.pop()
            for word in evaluate_all_words(board, word_lengths[-1], solution_word, EXCLUDE_SET):

                if 'cufflink' in solution and 'saw' in solution:
                    print(word, solution)

                # remove word from board
                board_copy = remove_word(np.copy(board), word.get_path())
                iters += 1

                if (board_copy == EMPTY_CELL).all():
                    if _is_solution_valid(solution + [word], EXCLUDE_SET, SOLUTION_SET):
                        print('SOLVED! ', solution + [word])
                        results.append(solution + [word])

                if len(word_lengths) > 1:
                    solution_offset = len(solution) + 1
                    solution_word = SOLUTION_DICT.get(solution_offset)
                    stack.append((board_copy, word_lengths[:-1], solution + [word], solution_word))

                # periodically purge the stack given our set of hints
                if IS_NEW_HINT or iters % 1000 == 0:
                    # stack = _purge_stack(stack, letter_count, word_lengths, EXCLUDE_SET, SOLUTION_SET)
                    stack = _purge_stack(stack, letter_count, EXCLUDE_SET, SOLUTION_SET)
                    results = _purge_results(results, letter_count, EXCLUDE_SET, SOLUTION_SET)

                    if IS_NEW_HINT:
                        IS_NEW_HINT = False

                # this is just here for logging so the user knows what's going on
                if iters % 1000 == 0:
                    e = time.time()
                    print('\nStack = {}\tIter = {}\tTime = {}\n'.format(len(stack), iters, e - s))
                    s = time.time()
                    print('Solution set: ', SOLUTION_SET)
                    print('Exclude set: ', EXCLUDE_SET)
                    print("Current solution: ", solution + [word])
                    print('-------')
    except KeyboardInterrupt:
        print('Exiting and printing solutions...')

    return results


def solve_game(board, word_lengths):
    height, width = board.shape

    total_cells = height * width
    assert total_cells >= sum(word_lengths)

    word_lengths.reverse()
    return solve_board(board, word_lengths)


def load_words():
    print('loading english words...')
    s = time.time()
    trie_files = futil.get_filepaths(trie.TRIE_FILE_PATH, ext=trie.TRIE_EXT)
    tries = trie.load_tries(trie_files)
    e = time.time()
    print('Done loading words... {}s'.format(e - s))
    return tries


def process_image(image_name):
    s = time.time()
    board, solution_word_lens = image_utils.get_puzzle_params(image_name)
    e = time.time()
    print('Extracted board in {}s'.format(e - s))
    print(board)
    print(solution_word_lens)

    return board, solution_word_lens    


def file_reader():
    """
    Read from hint file and update EXCLUDE and SOLUTION sets
    This is excuted in a separate thread
    :return:
    """
    prev_exclude_len = 0
    prev_solution_len = 0

    if not futil.is_file(HINT_FILE):
        futil.touch(HINT_FILE)

    while True:
        with open(HINT_FILE, 'r') as fin:
            for line in fin:
                line = line.lower().strip()
                vs = line.split(':')
                if len(vs) != 2:
                    continue

                if vs[0] == 'x':
                    EXCLUDE_SET.add(vs[1])
                elif vs[0] == 'i':
                    SOLUTION_SET.add(vs[1])
                elif vs[0].isdigit:
                    SOLUTION_SET.add(vs[1])
                    SOLUTION_DICT[int(vs[0]) - 1] = vs[1]

        if len(EXCLUDE_SET) > prev_exclude_len or len(SOLUTION_SET) > prev_solution_len:
            # something changed!
            # TODO: could use this to send a signal to the main thread
            IS_NEW_HINT = True

        prev_exclude_len = len(EXCLUDE_SET)
        prev_solution_len = len(SOLUTION_SET)
        time.sleep(2)


if __name__ == '__main__':

    TRIES = load_words()

    #
    # File path from which to fetch an image of the board
    #
    image_path = '/Users/nlarusso/Dropbox (Personal)/Camera Uploads/'

    # Grab the latest image from this folder and parse it
    image_name = futil.find_latest_image(image_path)
    board, solution_word_lens = process_image(image_name)

    # kick off the extra thread to read in any hints from the user
    t = threading.Thread(target=file_reader)
    t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
    t.start()

    #
    # Start solving the puzzle!!!
    #
    s = time.time()
    results = solve_game(board, solution_word_lens)
    e = time.time()
    print('{} seconds total'.format(e - s))

    #
    # Print the possible solutions with some pretty formatting!
    #
    res_words = {}
    for words in results:
        w = '-'.join([str(word) for word in words])
        if w not in res_words:
            res_words[w] = words
            print('\nSolution: ')
            for w1 in words:
                print('\t{:10}, {}'.format(w1.get_word(), w1.get_path()))

