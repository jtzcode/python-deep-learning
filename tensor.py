import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

x = np.array([[5, 78, 2, 34, 0],
            [6, 79, 3, 35, 1],
            [7, 80, 4, 36, 2]])
print(x.ndim)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.dtype)

digit = train_images[4]
plt.imshow(digit, cmap='binary')
plt.show()