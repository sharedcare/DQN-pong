#!/usr/bin/python
from __future__ import print_function
import numpy as np
import pygame
import random
import collections
import sys
from sys import argv

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)

class PongGame(object):

    def __init__(self):
        pygame.init()
        self.FPS = 60
        self.QFPS = 240
        self.GAME_PADDING = 20
        self.GAME_WIDTH, self.GAME_HEIGHT = 800, 600
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 8
        self.MAX_SCORE_PER_GAME = 10
        self.BALL_RADIUS = 10
        self.ball_pos = [0, 0]      # range from [0, 0] to [GAME_WIDTH, GAME_HEIGHT]
        self.ball_vel = [0, 0]      # can be negative number
        self.paddle1_vel = 20        # must be non-negative number
        self.paddle2_vel = 20        # must be non-negative number
        self.score = [0, 0]

    def reset_game(self):
        self.frames = collections.deque(maxlen=10)
        self.game_over = False

        self.reset_ball(win_side=random.randint(1,2))
        self.score = [0, 0]
        self.reward = [0, 0]
        self.SCREEN = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        self.CLOCK = pygame.time.Clock()

    def reset_ball(self, win_side):
        self.paddle1_x = self.GAME_WIDTH // 2
        self.paddle2_x = self.GAME_WIDTH // 2
        self.ball_pos = [self.GAME_WIDTH // 2, self.GAME_HEIGHT // 2]
        vel_h = random.randrange(-8, 8)
        vel_v = random.randrange(8, 10)
        if win_side == 1:
            self.ball_vel[1] = vel_v
        elif win_side == 2:
            self.ball_vel[1] = -vel_v

        self.ball_vel[0] = vel_h

    def step(self, action):
        pygame.event.pump()

        if action[0] == 0:      # player1 move paddle left
            self.paddle1_x -= self.paddle1_vel if (self.paddle1_x - self.paddle1_vel > 0) else 0

        elif action[0] == 1:    # player1 move paddle right
            self.paddle1_x += self.paddle1_vel if (
                self.paddle1_x + self.paddle1_vel < self.GAME_WIDTH) else self.GAME_WIDTH

        elif action[0] == 2:                   # player1 do nothing
            pass

        if action[1] == 0:      # player2 move paddle left
            self.paddle2_x -= self.paddle2_vel if (self.paddle2_x - self.paddle2_vel > 0) else 0

        elif action[1] == 1:    # player2 move paddle right
            self.paddle2_x += self.paddle2_vel if (
                self.paddle2_x + self.paddle2_vel < self.GAME_WIDTH) else self.GAME_WIDTH

        elif action[1] == 2:                   # player2 do nothing
            pass

        self.SCREEN.fill(BLACK)

        # update paddles position
        paddle1 = pygame.draw.rect(self.SCREEN, WHITE,
                                   pygame.Rect(self.paddle1_x,
                                              (self.GAME_HEIGHT - self.GAME_PADDING),
                                               self.PADDLE_WIDTH,
                                               self.PADDLE_HEIGHT))

        paddle2 = pygame.draw.rect(self.SCREEN, WHITE,
                                   pygame.Rect(self.paddle2_x,
                                               self.GAME_PADDING,
                                               self.PADDLE_WIDTH,
                                               self.PADDLE_HEIGHT))

        self.reward = [0, 0]
        # update ball position
        self.ball_pos[0] += int(self.ball_vel[0])
        self.ball_pos[1] += int(self.ball_vel[1])
        ball = pygame.draw.circle(self.SCREEN, RED, self.ball_pos, self.BALL_RADIUS)

        # check ball collision on left and right walls
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_vel[0] = -self.ball_vel[0]
        if self.ball_pos[0] >= self.GAME_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] = -self.ball_vel[0]

        # check ball collision on walls or paddles
        if self.ball_pos[1] > (self.GAME_HEIGHT - self.GAME_PADDING) - self.BALL_RADIUS:
            self.reward = [-1, 1]
            self.score[1] += 1
            self.reset_ball(2)
        elif (self.GAME_HEIGHT - self.GAME_PADDING) - self.BALL_RADIUS - self.PADDLE_HEIGHT \
                <= self.ball_pos[1] <= (self.GAME_HEIGHT - self.GAME_PADDING) - self.BALL_RADIUS and \
                self.paddle1_x - (self.PADDLE_WIDTH / 2) <= self.ball_pos[0] <= self.paddle1_x + (
                self.PADDLE_WIDTH / 2):     # ball collide on paddle1
                self.reward[0] = 0
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_vel[0] *= random.randrange(100, 120)/100
                self.ball_vel[1] *= random.randrange(100, 120)/100

        if self.ball_pos[1] < self.GAME_PADDING + self.BALL_RADIUS:
            self.reward = [1, -1]
            self.score[0] += 1
            self.reset_ball(1)
        elif self.GAME_PADDING + self.BALL_RADIUS + self.PADDLE_HEIGHT  \
                >= self.ball_pos[1] >= self.GAME_PADDING + self.BALL_RADIUS and \
                self.paddle2_x - (self.PADDLE_WIDTH / 2) <= self.ball_pos[0] <= self.paddle2_x + (
                self.PADDLE_WIDTH / 2):     # ball collide on paddle2
                self.reward[1] = 0
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_vel[0] *= random.randrange(100, 120)/100
                self.ball_vel[1] *= random.randrange(100, 120)/100

        print(pygame.surfarray.array3d(pygame.display.get_surface()))

        # save last 10 frames
        # a frame is decided by ball_x, ball_y, and paddle_x
        self.frames.append([self.ball_pos[0], self.ball_pos[1], self.paddle1_x, self.paddle2_x])

        if (self.score[0] >= self.MAX_SCORE_PER_GAME) or (self.score[1] >= self.MAX_SCORE_PER_GAME):
            self.game_over = True

        pygame.display.flip()
        self.CLOCK.tick(self.FPS)
        return self.get_frames(), self.reward, self.game_over

    def get_frames(self):
        return np.array(list(self.frames))


def main(_):
    pong_game = PongGame()
    NUM_EPOCHS = 5
    for e in range(NUM_EPOCHS):
        print("Epoch: {:d}".format(e))
        pong_game.reset_game()
        # input_t = game.get_frames()
        game_over = False
        while not game_over:
            # pick a random action
            action = [np.random.randint(0, 3), np.random.randint(0, 3)]
            last4frames, reward, game_over = pong_game.step(action)
            sys.stdout.write("\rplayer1: action = %d, reward = %2d" % (action[0], reward[0]))
        print("\nplayer1_score = {0}, player2_score = {1}".format(pong_game.score[0], pong_game.score[1]))


if __name__ == '__main__':
    main(argv)
