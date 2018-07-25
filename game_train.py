#!/usr/bin/python
from __future__ import print_function
from pong import PongGame
import tensorflow as tf
import collections
import numpy as np
import argparse
import sys
import os


def preprocess_frames(frames):
    if frames.shape[0] < 4:
        # single frame
        x_t = frames[0].astype("float")
        x_t /= 80.0
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=1)
        # s_t.shape = (3, 4), duplicate x_t 4 times.
    else:
        # 4 frames
        xt_list = []
        for i in range(4): # frames.shape[0]):
            x_t = frames[i].astype("float")
            x_t /= 80.0
            xt_list.append(x_t)
        s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]), axis=1)
    s_t = np.expand_dims(s_t, axis=2)
    s_t = np.expand_dims(s_t, axis=0)
    # s_t.shape = (1, 3, 4, 1)
    return s_t

def train(game, num_iteration):
    for e in range(num_iteration):
        loss = 0.0
        game.reset_game()

        # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over = game.step(a_0)
        s_t = preprocess_frames(x_t)

        while not game_over:
            pass

def main(_):
    # initialize parameters
    DATA_DIR = ""
    NUM_ACTIONS = 3 # number of valid actions (left, stay, right)
    GAMMA = 0.99 # decay rate of past observations
    INITIAL_EPSILON = 0.1 # starting value of epsilon
    FINAL_EPSILON = 0.0001 # final value of epsilon
    MEMORY_SIZE = 50000 # number of previous transitions to remember
    NUM_EPOCHS_OBSERVE = 1
    NUM_EPOCHS_TRAIN = 1000

    BATCH_SIZE = 1
    NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

    model_file = "pong_net"

    if os.path.exists(model_file):
        # load the model
        pass
    else:
        # build the model
        pass

    pong_game = PongGame()
    experience = collections.deque(maxlen=MEMORY_SIZE)

    num_games, num_wins = 0, 0
    epsilon = INITIAL_EPSILON

    for e in range(NUM_EPOCHS):
        loss = 0.0
        pong_game.reset_game()

        # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over = pong_game.step(a_0)
        s_t = preprocess_frames(x_t)

        while not game_over:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='./nsfw_model/open_nsfw.pb',
        help='Path to NSFW classification model'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Absolute path to image file'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)