import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Init Network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compile
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare Image Data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# Prepare Label Data
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fitting (Training)
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluation
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)