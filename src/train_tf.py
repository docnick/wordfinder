from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import file_utils as util
import time
import os


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
LETTER_LOOKUP = {}
REV_LETTER_LOOKUP = {}
TRAIN_DATA_PATH = 'puzzle'

MODEL_VERSION = 1.0
MODEL_NAME = 'letter_class-{}.tensor'.format(MODEL_VERSION)


def build_letter_lookup():
    for i, letter in enumerate(list(ALPHABET)):
        res_vec = np.zeros(len(ALPHABET), dtype=np.float32)
        res_vec[i] = 1
        LETTER_LOOKUP[letter] = res_vec
        REV_LETTER_LOOKUP[np.argmax(res_vec)] = letter


def convert_image_to_vec(image):
    width, height = image.size
    vec = []

    data = image.load()
    for i in range(width):
        for j in range(height):
            vec.append(data[i, j] / 255.0)

    return np.array(vec, dtype=np.float32)


def build_data(images):

    data = []
    data_class = []
    for i, image_file in enumerate(images):
        im = Image.open(image_file)
        img_vec = convert_image_to_vec(im)

        file_name = os.path.split(image_file)[1]
        answer = LETTER_LOOKUP.get(file_name[0])

        data.append(img_vec)
        data_class.append(answer)

    return np.array(data), np.array(data_class)


def build_model(pixel_count, answer_size):
    # input image
    x = tf.placeholder(tf.float32, [None, pixel_count], name='x')
    # hidden layer
    W = tf.Variable(tf.zeros([pixel_count, answer_size]), name='W')
    # bias
    b = tf.Variable(tf.zeros([answer_size]), name='b')
    # answer output
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

    y_ = tf.placeholder(tf.float32, [None, answer_size], name='y_')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

    return x, y, y_, train_step


def classify_letter(letters):
    build_letter_lookup()
    tf.reset_default_graph()

    with tf.Session() as sess:
        x, y, y_, train_step = build_model(len(letters[0]), len(ALPHABET))
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        saver.restore(sess, MODEL_NAME)

        letter_strs = []
        for letter in letters:
            classification = sess.run(tf.argmax(y, 1), feed_dict={x: [letter]})
            letter_strs.append(REV_LETTER_LOOKUP.get(classification[0]))

    return letter_strs


def train(data, data_class):
    x, y, y_, train_step = build_model(len(data[0]), len(data_class[0]))
    with tf.Session() as sess:
        print('Starting to train model...')
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        s = time.time()
        for i in range(300):
            train_step.run({x: data, y_: data_class})

            if i % 50 == 0:
                print('iteration = {}'.format(i))
                samples = np.random.randint(0, len(data), 10)
                acc = 0
                for i in samples:
                    d = data[i]
                    c = data_class[i]
                    classification = sess.run(tf.argmax(y, 1), feed_dict={x: [d]})
                    acc += classification[0] == np.argmax(c)
                print('accuracy: {}'.format(acc / 10.0))

        e = time.time()
        print('model trained in {} s'.format(e - s))

        saver.save(sess, MODEL_NAME)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy of trained model')
        print(accuracy.eval({x: data, y_: data_class}))

        # print('RESULTS')
        # for d, c in zip(data, data_class):
        #     classification = sess.run(tf.argmax(y, 1), feed_dict={x: [d]})
        #     print(classification[0], np.argmax(c))


def test_model(data, data_class):
    tf.reset_default_graph()

    with tf.Session() as sess:
        x, y, y_, train_step = build_model(len(data[0]), len(data_class[0]))
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        for v in tf.all_variables():
            print(v.op.name)

        print('restoring model...')
        saver.restore(sess, MODEL_NAME)

        for v in tf.all_variables():
            print(v.op.name)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy of trained model')
        print(accuracy.eval({x: data, y_: data_class}))

    # print('Test the trained model...')
    # acc = 0
    # for d, c in zip(data, data_class):
    #     classification = sess.run(tf.argmax(y, 1), feed_dict={x: [d]})
    #     print(classification[0], np.argmax(c))
    #     acc += classification[0] == np.argmax(c)
    # print('accuracy: {}'.format(acc / (1.0 * len(data))))


if __name__ == '__main__':
    build_letter_lookup()
    letter_images = util.get_filepaths(TRAIN_DATA_PATH, ext='png')
    print('Found {} images'.format(len(letter_images)))

    data, data_class = build_data(letter_images)
    train(data, data_class)
    test_model(data, data_class)
