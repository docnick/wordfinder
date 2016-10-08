## WordFinder
Automated solver for [wordbrain](https://play.google.com/store/apps/details?id=se.maginteractive.wordbrain&hl=en).

This code will:
* Parse an image of a wordbrain board extracting the letters and sizes of the solution words
* Iterate through all possible solutions of the board providing the words and paths 
* Read from a file (in a separate thread) to allow the user to enter hints about words that are or are not in the solution 


## Getting started:
You'll need to do a few things before using the code:

1. Install the requirements. I developed and tested python 3.5 `mkvirtualenv --python=/usr/local/bin/python3.5 wordfinder3`, though everything should work for 2.7 as well.
    `> pip install -r requirements.txt`
2. Build the Tries: these are lookups on valid English words.
    
    0. Make sure you have your word list: `word_data/words.dat`
    1. Run `trie/config_tries.py`, this will generate a sequence of trie files under `word_data/*.trie`. Each trie contains all words with length at least K. E.g. the trie file en_words-4.trie contains all words with a length greater then 3.
3. Train the letter recognition model
    
    0. Make sure you have images and correct parsings in data/ (you'll need to bunzip and untar `data.tar.bzip` first) 
    1. run `letter_extractor.py`, this will parse the images and create the training data required by tensorflow
    2. run `train.tf.py`, this pulls in the data created in the previous step, trains the tensorflow model for classifying letter images into characters and saves the model to a file.


## Running the program
Run `python find_words.py`, this will parse the image, extract the board, identify word lengths for the solution, and solve the puzzle


## TODO:
* Performance!!! 
* clean up code
* add logging
* Ranking solutions: examine which words were used in previous puzzles and use that to weight solutions
* http://pythonhosted.org/pythran/