# utils/model_utils.py

import tensorflow as tf

def build_generator(latent_dim):
    """
    Build the generator model.
    """
    from models.generator import Generator
    inputs = tf.keras.Input(shape=(latent_dim,))
    generator = Generator(latent_dim)
    outputs = generator(inputs)
    return tf.keras.Model(inputs, outputs, name="generator")

def build_discriminator(input_dim):
    """
    Build the discriminator model.
    """
    from models.discriminator import Discriminator
    inputs = tf.keras.Input(shape=(input_dim,))
    discriminator = Discriminator()
    outputs = discriminator(inputs)
    return tf.keras.Model(inputs, outputs, name="discriminator")

def save_model(model, file_path):
    """
    Save the model to a file.
    """
    model.save(file_path)

def load_model(file_path):
    """
    Load a model from a file.
    """
    return tf.keras.models.load_model(file_path)

