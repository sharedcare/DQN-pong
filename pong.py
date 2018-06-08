#!/usr/bin/python
from __future__ import print_function
import pygame
import random
import collections
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
        self.GAME_WIDTH, self.GAME_HEIGHT = 640, 480
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 64, 8
        self.BALL_RADIUS = 10
        self.ball_pos = [0, 0]      # range from [0, 0] to [GAME_WIDTH, GAME_HEIGHT]
        self.ball_vel = [0, 0]      # can be negative number
        self.paddle1_vel = 0        # must be non-negative number
        self.paddle2_vel = 0        # must be non-negative number
        self.score = [0, 0]

    def reset_game(self):
        self.frames = collections.deque(maxlen=4)
        self.game_over = False

        self.reset_ball(win_side=random.randint(1,2))
        self.score = [0, 0]
        self.reward = 0
        self.SCREEN = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        self.CLOCK = pygame.time.Clock()

    def reset_ball(self, win_side):
        self.paddle1_x = self.GAME_WIDTH / 2
        self.paddle2_x = self.GAME_WIDTH / 2
        self.ball_pos = [self.GAME_WIDTH / 2, self.GAME_HEIGHT / 2]
        vel_h = random.randrange(-3, 3)
        vel_v = random.randrange(2, 4)
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

        self.reward = 0
        # update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball = pygame.draw.circle(self.SCREEN, RED, self.ball_pos, self.BALL_RADIUS)

        # check ball collision on left and right walls
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_vel[0] = -self.ball_vel[0]
        if self.ball_pos[0] >= self.GAME_HEIGHT - self.BALL_RADIUS:
            self.ball_vel[0] = -self.ball_vel[0]

        # check ball collision on walls or paddles
        if self.ball_pos[1] >= (self.GAME_HEIGHT - self.GAME_PADDING) - self.BALL_RADIUS:
            if self.paddle1_x - (self.PADDLE_WIDTH / 2) <= self.ball_pos[0] <= self.paddle1_x + (
                        self.PADDLE_WIDTH / 2):    # ball collide on paddle1
                self.reward = 0
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_vel[0] *= 1.1
                self.ball_vel[1] *= 1.1
            else:
                self.reward = -1
                self.score[1] += 1

        if self.ball_pos[1] <= self.GAME_PADDING + self.BALL_RADIUS:
            if self.paddle2_x - (self.PADDLE_WIDTH / 2) <= self.ball_pos[0] <= self.paddle2_x + (
                        self.PADDLE_WIDTH / 2):    # ball collide on paddle2
                self.reward = 0
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_vel[0] *= 1.1
                self.ball_vel[1] *= 1.1
            else:
                self.reward = 1
                self.score[0] += 1


def main(_):
    pong_game = PongGame()
    pong_game.reset_game()


if __name__ == '__main__':
    main(argv)
