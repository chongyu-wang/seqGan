# models/discriminator.py

import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = layers.Dense(1024, activation='leaky_relu')
        self.dense2 = layers.Dense(512, activation='leaky_relu')
        self.dense3 = layers.Dense(256, activation='leaky_relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

def build_discriminator(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    discriminator = Discriminator()
    outputs = discriminator(inputs)
    return tf.keras.Model(inputs, outputs, name="discriminator")
