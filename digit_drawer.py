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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                if(CANVAS_X <= mouse_x <= CANVAS_X + CANVAS_SIZE and CANVAS_Y <= mouse_y <= CANVAS_Y + CANVAS_SIZE):
                    drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mouse_x, mouse_y = event.pos
                canvas_x = mouse_x - CANVAS_X
                canvas_y = mouse_y - CANVAS_Y

                if 0 <= canvas_x < CANVAS_SIZE and 0 <= canvas_y < CANVAS_SIZE:
                    pygame.draw.circle(canvas, WHITE, (canvas_x, canvas_y), 8)

    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (CANVAS_X-2, CANVAS_Y-2, CANVAS_SIZE+4, CANVAS_SIZE+4), 2)
    screen.blit(canvas, (CANVAS_X, CANVAS_Y))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

