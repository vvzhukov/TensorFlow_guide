# TensorFlow 2.15 recap guide
# Based on official guide from Google with extra comments
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

# IMDB dataset
# 50,000 movie reviews. 25k for training and 25k for testing.

# 15k - training; 10k - validation; 25k - testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Explore data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print("Train example:", train_examples_batch)
print("First 10 labels:", train_labels_batch)

# Pre-trained text embedding as the first layer
# Why?
#   You don't have to worry about text preprocessing,
#   Benefit from transfer learning,
#   The embedding has a fixed size, so it's simpler to process.

# Will use pre-trained text embedding model from TensorFlow Hub 'google/nnlm-en-dim50/2'
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

# Build the model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

# Layers are stacked sequentially to build the classifier
#
# The first layer is a TensorFlow Hub layer (Pre-trained Saved Model to map a sentence into its embedding vector)
# Second - fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
# The last layer is densely connected with a single output node.

# Loss function and optimizer
# Binary classification problem and the model outputs logits -> binary_crossentropy (not a single option).
#   Generally, binary_crossentropy is better for dealing with probabilities—it measures the "distance"
#   between probability distributions, or in our case, between the ground-truth distribution and the predictions.

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Model evaluation
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

# MIT License
#
# Copyright (c) 2017 François Chollet
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