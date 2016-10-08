import numpy as np


def parse_board(board):
    rows = board.split('\n')

    mat = []
    for row in rows:
        if not row:
            continue

        row = list(row)
        mat.append(row)

    return np.matrix(mat)
