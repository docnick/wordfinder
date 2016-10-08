import os


def get_filepaths(directory, ext=None):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if ext is not None and filename[-len(ext):] == ext or ext is None:
                file_paths.append(filepath)  # Add it to the list.

    return file_paths


def create_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def find_latest_image(directory):
    files = get_filepaths(directory, ext='png')
    latest = max(files, key=os.path.getctime)
    print('using image: {}'.format(latest))
    return latest


def is_file(filename):
    if os.path.isfile(filename):
        return True

    return False


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)