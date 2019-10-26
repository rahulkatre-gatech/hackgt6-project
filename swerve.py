import pygame

# Screen
WIDTH = 1280
HEIGHT = 960
SCREEN_SIZE = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Pong')
clock = pygame.time.Clock()
pygame.key.set_repeat(50, 50)
pygame.init()

# Box
box_width = 30
box_height = 30
max_vel = 30  # pix/sec

while True:
    pygame.rect.Rect


pygame.quit()
