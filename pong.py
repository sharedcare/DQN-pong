#!/usr/bin/python
import pygame
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
        self.GAME_WIDTH, self.GAME_HEIGHT = 640, 480
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 64, 8
        self.BALL_RADIUS = 10
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.score = [0, 0]

    def reset(self):
        self.frames = collections.deque(maxlen=4)
        self.game_over = False

        self.paddle_x = self.GAME_WIDTH // 2
        self.score = [0, 0]
        self.reward = 0
        self.ball_pos = [self.GAME_WIDTH // 2, self.GAME_HEIGHT // 2]
        self.SCREEN = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        self.CLOCK = pygame.time.Clock()

def main(_):
    pong_game = PongGame()
    pong_game.reset()

if __name__ == '__main__':
    main(argv)