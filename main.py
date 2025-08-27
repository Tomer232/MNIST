from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

#load the MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

#print the first 4 pictures in a row
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

