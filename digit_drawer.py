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

CLEAR_BUTTON_X = PREDICTION_X + 10
CLEAR_BUTTON_Y = PREDICTION_Y + 220
CLEAR_BUTTON_WIDTH = 80
CLEAR_BUTTON_HEIGHT = 30

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("MNIST Digit Recognizer")

font_large = pygame.font.Font(None, 72)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)

print("Loading model...")
model = tf.keras.models.load_model('mnist_model.h5')
print("Model loaded successfully")


def predict_digit(canvas_surface):
    canvas_array = pygame.surfarray.array3d(canvas_surface)
    canvas_array = np.transpose(canvas_array, (1, 0, 2))
    gray_array = np.dot(canvas_array[..., :3], [0.299, 0.587, 0.114])

    img = Image.fromarray(gray_array.astype('uint8'))
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    predictions = model.predict(img_array, verbose=0)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_digit = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100

    return predicted_digit, confidence


canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(BLACK)

drawing = False
clock = pygame.time.Clock()

current_prediction = "?"
current_confidence = "0.0%"

# main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                if CANVAS_X <= mouse_x <= CANVAS_X + CANVAS_SIZE and CANVAS_Y <= mouse_y <= CANVAS_Y + CANVAS_SIZE:
                    drawing = True
                    prev_pos = (mouse_x - CANVAS_X, mouse_y - CANVAS_Y)
                elif CLEAR_BUTTON_X <= mouse_x <= CLEAR_BUTTON_X + CLEAR_BUTTON_WIDTH and CLEAR_BUTTON_Y <= mouse_y <= CLEAR_BUTTON_Y + CLEAR_BUTTON_HEIGHT:
                    canvas.fill(BLACK)
                    current_prediction = "?"
                    current_confidence = "0.0%"
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                canvas_array = pygame.surfarray.array3d(canvas)
                if np.any(canvas_array > 0):
                    pred_digit, confidence = predict_digit(canvas)
                    current_prediction = str(pred_digit)
                    current_confidence = f"{confidence:.1f}%"
                else:
                    current_prediction = "?"
                    current_confidence = "0.0%"
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mouse_x, mouse_y = event.pos
                canvas_x = mouse_x - CANVAS_X
                canvas_y = mouse_y - CANVAS_Y

                if 0 <= canvas_x < CANVAS_SIZE and 0 <= canvas_y < CANVAS_SIZE:
                    if 'prev_pos' in locals():
                        pygame.draw.line(canvas, WHITE, prev_pos, (canvas_x, canvas_y), 16)
                    else:
                        pygame.draw.circle(canvas, WHITE, (canvas_x, canvas_y), 8)
                    prev_pos = (canvas_x, canvas_y)

    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (CANVAS_X - 2, CANVAS_Y - 2, CANVAS_SIZE + 4, CANVAS_SIZE + 4), 2)
    screen.blit(canvas, (CANVAS_X, CANVAS_Y))

    pygame.draw.rect(screen, GRAY, (PREDICTION_X, PREDICTION_Y, 180, 100), 2)
    prediction_text = font_small.render("Predicted Digit:", True, BLACK)
    screen.blit(prediction_text, (PREDICTION_X + 10, PREDICTION_Y + 10))

    pygame.draw.rect(screen, GRAY, (PREDICTION_X, PREDICTION_Y + 120, 180, 80), 2)
    conf_text = font_small.render("confidence:", True, BLACK)
    screen.blit(conf_text, (PREDICTION_X + 10, PREDICTION_Y + 130))

    digit_display = font_large.render(current_prediction, True, BLACK)
    screen.blit(digit_display, (PREDICTION_X + 80, PREDICTION_Y + 35))

    confidence_display = font_medium.render(current_confidence, True, BLACK)
    screen.blit(confidence_display, (PREDICTION_X + 80, PREDICTION_Y + 155))

    pygame.draw.rect(screen, GRAY, (CLEAR_BUTTON_X, CLEAR_BUTTON_Y, CLEAR_BUTTON_WIDTH, CLEAR_BUTTON_HEIGHT))
    pygame.draw.rect(screen, BLACK, (CLEAR_BUTTON_X, CLEAR_BUTTON_Y, CLEAR_BUTTON_WIDTH, CLEAR_BUTTON_HEIGHT), 2)
    clear_text = font_small.render("Clear", True, BLACK)
    text_rect = clear_text.get_rect(
        center=(CLEAR_BUTTON_X + CLEAR_BUTTON_WIDTH // 2, CLEAR_BUTTON_Y + CLEAR_BUTTON_HEIGHT // 2))
    screen.blit(clear_text, text_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
