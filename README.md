
## Build the Tries
0) Make sure you have your word list: `word_data/words.dat`
1) Run `trie/config_tries.py`, this will generate a sequence of trie files under `word_data/*.trie`


## Training letter recognition
0) Make sure you have images and correct parsings in data/ 
1) run `letter_extractor.py`, this will parse the images and create the training data required by tensorflow
2) run `train.tf.py`, this pulls in the data created in the previous step, trains the tensorflow model for classifying letter images into characters and saves the model to a file.


## Running the program
1) run `find_words.py`, this will parse the image, extract the board, identify word lengths for the solution, and solve the puzzle


## TODO:
* Performance!!! 
* clean up code
* add logging
* Ranking solutions: examine which words were used in previous puzzles and use that to weight solutions
* http://pythonhosted.org/pythran/
