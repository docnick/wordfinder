from PIL import Image
import os
from collections import defaultdict

import file_utils as utils
import image_utils
import board_utils


DATA_DIR = '../data'
PUZZLE_PATH = 'puzzle'

LETTERS_SEEN = defaultdict(int)


def save_letters(board, puzzle_path, answers):
    for letter in image_utils.extract_letters(board, answers):
        image = letter.get('image')
        full_path = os.path.join(puzzle_path, '{}_{}-{}.png'.format(letter.get('class'),
                                                                    letter.get('x'),
                                                                    letter.get('y')))
        image.save(full_path)
        LETTERS_SEEN[letter.get('class')] += 1


def create_training_folder(image_file):
    utils.create_folder(PUZZLE_PATH)

    board_name = os.path.split(image_file)[-1]
    puzzle_path = os.path.join(PUZZLE_PATH, board_name[:-4])

    utils.create_folder(puzzle_path)

    return puzzle_path


def get_board(file_name):
    board = []
    with open(file_name, 'r') as fin:
        for line in fin:
            board.append(line.strip())

    return '\n'.join(board)


if __name__ == '__main__':

    images = utils.get_filepaths(DATA_DIR, ext='png')

    for i, image_file in enumerate(images):
        board_str = get_board(image_file[:-4] + '.txt')
        answers = board_utils.parse_board(board_str)

        puzzle_path = create_training_folder(image_file)
        print('puzzle path: {}'.format(puzzle_path))

        im = Image.open(image_file)
        bw_im = image_utils.convert_to_bw(im)

        board, solution = image_utils.parse_image(bw_im)
        print(image_utils.get_solution_word_lengths(solution))

        save_letters(board, puzzle_path, answers)

    print('\nLetters observed [{}]...'.format(len(LETTERS_SEEN)))
    for letter in sorted(list(LETTERS_SEEN.keys())):
        print('{} : {}'.format(letter, LETTERS_SEEN.get(letter)))
