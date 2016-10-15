from PIL import Image, ImageFilter, ImageOps
import numpy as np
import train_tf as tf

# ########################################################################
# The code for figuring out where the board starts/ends and where the solution
# blocks start/end is pretty hacky. I've only tested on a samsung s5, so
# if the board images are different sizes on other devices, these methods
# will likely break.
# ########################################################################


CLASS_IMG_SIZE = (32, 32)
START_ROW = 340
END_ROW = 1400
PIXEL_THRESH = 500


def convert_to_bw(image):
    """convert a color image to black and white"""
    gray = image.convert('L')
    bw_image = gray.point(lambda x: 0 if x < 128 else 255, '1')
    return bw_image


def sum_pixel_cols(image):
    data = image.load()
    width, height = image.size

    col_white_pixel_counts = []
    for j in range(width):
        s = 0
        for i in range(height):
            if data[j, i] == 255:
                s += 1
        col_white_pixel_counts.append(s)

    return np.array(col_white_pixel_counts)


def sum_pixel_rows(image):
    data = image.load()
    width, height = image.size

    row_white_pixel_counts = []
    for i in range(height):
        s = 0
        for j in range(width):
            if data[j, i] == 255:
                s += 1
        row_white_pixel_counts.append(s)

    return np.array(row_white_pixel_counts)


def _get_edges(vals, pixel_thresh=100):
    noise = 1

    edges = []
    prev_pixel = vals[0]
    start_block = None
    for i in range(2, len(vals)):
        if prev_pixel <= noise and abs(vals[i] - prev_pixel) >= pixel_thresh:
            start_block = i
        elif start_block is not None and vals[i] <= noise and abs(prev_pixel - vals[i]) > pixel_thresh:
            edges.append((start_block, i))
            start_block = None

        prev_pixel = vals[i - 1]

    return edges


def _find_edge(rows, pixel_thresh, start, stop):
    edge_row = None

    last = rows[start - 1]
    for i in range(start, stop):
        if abs(rows[i] - last) >= pixel_thresh:
            edge_row = i
            break
        last = rows[i]

    return edge_row


def get_solution_rows(solution):
    width, height = solution.size
    rows = sum_pixel_rows(solution)
    edges = _get_edges(rows)

    solution_rows = []
    for row in edges:
        solution_rows.append(solution.crop((0, row[0], width, row[1])))

    return solution_rows


def get_word_lengths(solution_row):
    cols = sum_pixel_cols(solution_row)

    edges = _get_edges(cols, pixel_thresh=15)
    prev = edges[0]
    distances = []
    for square in edges[1:]:
        distances.append(square[0] - prev[1])
        prev = square

    med = np.median(distances)

    word_lengths = []
    word_len = 1
    for d in distances:
        if d > med + 5:
            word_lengths.append(word_len)
            word_len = 1
        else:
            word_len += 1

    if word_len > 1:
        word_lengths.append(word_len)

    return word_lengths


def get_solution_word_lengths(solution):

    word_lengths = []
    for solution_row in get_solution_rows(solution):
        wlen = get_word_lengths(solution_row)
        word_lengths.extend(wlen)

    return word_lengths


def parse_image(image):
    width, height = image.size
    rows = sum_pixel_rows(image)

    # TODO: test these offsets on pictures taken from other devices
    start_row = _find_edge(rows, PIXEL_THRESH, START_ROW, START_ROW + 50)
    end_row = _find_edge(rows, PIXEL_THRESH, END_ROW, END_ROW + 50)

    if start_row is None or end_row is None:
        print('ERROR!')
        print('start_row = {}\tend_row = {}'.format(start_row, end_row))
        print(rows)

    board = image.crop((0, start_row - 5, width, end_row + 5))

    solution_start_row = _find_edge(rows, 250, end_row + 50, end_row + 150)
    rev_rows = rows[::-1]
    solution_end_row = _find_edge(rev_rows, 100, 180, 400)

    solution = image.crop((0, solution_start_row - 5, width, height - solution_end_row + 5))

    return board, solution


def extract_letters(board, classification=None):
    buffer = 25
    rows = sum_pixel_rows(board)
    cols = sum_pixel_cols(board)

    row_edges = _get_edges(rows)
    col_edges = _get_edges(cols)

    letters = []
    for i, cpos in enumerate(col_edges):
        for j, rpos in enumerate(row_edges):

            letter = board.crop((rpos[0] + buffer, cpos[0] + buffer, rpos[1] - buffer, cpos[1] - buffer))
            letter.thumbnail(CLASS_IMG_SIZE, Image.ANTIALIAS)

            if letter.size != CLASS_IMG_SIZE:
                print('image not correct size, fixing...')
                letter = ImageOps.fit(letter, (32, 32), Image.ANTIALIAS)
                print(letter.size)

            letter_obj = {'image': letter, 'x': i, 'y': j}
            if classification is not None:
                letter_obj['class'] = classification[i, j]

            letters.append(letter_obj)

    return letters


def get_puzzle_params(image_file):
    image = convert_to_bw(Image.open(image_file))

    board, solution = parse_image(image)
    letters = extract_letters(board)

    letter_vecs = []
    for letter in letters:
        letter_vecs.append(tf.convert_image_to_vec(letter.get('image')))

    letters = tf.classify_letter(letter_vecs)
    board = _construct_board(letters)

    solution_word_lengths = get_solution_word_lengths(solution)

    return board, solution_word_lengths


def _construct_board(letters):
    board_size = np.sqrt(len(letters))

    board = np.matrix(letters)
    board = board.reshape((board_size, board_size))
    return board
