import pygame
import numpy as np
import tensorflow as tf
from PIL import Image

CANVAS_SIZE = 280
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 350
CANVAS_X = 10
CANVAS_Y = 35
PREDICTION_X = 310
PREDICTION_Y = 50

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("MNIST Digit Recognizer")

print("Loading model...")
model = tf.keras.models.load_model('mnist_model.h5')
print("Model loaded successfully")

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(BLACK)

drawing = False
clock = pygame.time.Clock()

# main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (CANVAS_X-2, CANVAS_Y-2, CANVAS_SIZE+4, CANVAS_SIZE+4), 2)
    screen.blit(canvas, (CANVAS_X, CANVAS_Y))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

