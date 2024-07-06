# models/generator.py

import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(512, activation='relu')
        self.dense3 = layers.Dense(1024, activation='relu')
        self.output_layer = layers.Dense(8, activation='tanh')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_generator(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    generator = Generator(latent_dim)
    outputs = generator(inputs)
    return tf.keras.Model(inputs, outputs, name="generator")




