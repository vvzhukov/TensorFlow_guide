# TensorFlow 2.15 recap guide
# Based on official guide from Google with extra comments
# https://www.tensorflow.org/tutorials/quickstart/beginner

# Part 1. Train machine learning model using a prebuilt dataset using the Keras API.
#           Simplified version

import tensorflow as tf


# Check TF version
print("TensorFlow version:", tf.__version__)

# load MNIST data
# The MNIST database is a large database of handwritten digits that is commonly used
#   for training various image processing systems.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Rescale values from 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a model
# Sequential - a plain stack of layers where each layer has exactly one input tensor and one output tensor.
# Tensor - Tensors are multidimensional arrays with a uniform type.
# Layers - functions with a known mathematical structure that can be reused and have trainable variables.
#   Flatten - Flattens the input. Does not affect the batch size.
#   Dense - Just your regular densely-connected NN layer.
#   Dropout - Applies Dropout (randomly sets input units to 0 with a frequency of rate at each step during
#                               training time to the input.)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print("Vector of logits or log-odds scores, one for each class: ", predictions)

# Convert to probabilities:
print("Vector of probabilities:", tf.nn.softmax(predictions).numpy())
# Do not plug softmax function into the activation function for the last layer of the network
#   as it will impact the loss calculation

# Define the loss function.
#   Loss function takes a vector of ground truth values and a vector of logits and returns a
#       scalar loss for each example. This loss is equal to the negative log probability of
#       the true class: The loss is zero if the model is sure of the correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# This untrained model gives probabilities close to random (1/10 for each class),
#   so the initial loss should be close to 2.3 (tf.math.log(1/10))
print(loss_fn(y_train[:1], predictions).numpy())

# Training, configure and compile the model

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Adjust model parameters and minimize the loss
model.fit(x_train, y_train, epochs=5)

# Checks the model's performance on x_test set
model.evaluate(x_test,  y_test, verbose=2)

# To return a probability, wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.