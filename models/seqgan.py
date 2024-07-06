# models/seqgan.py

import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
import numpy as np

class SeqGAN(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(SeqGAN, self).__init__()
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator(input_dim)
        self.latent_dim = latent_dim

        # Optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(SeqGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            # Generate fake data using the generator
            generated_data = self.generator(random_latent_vectors)

            # Combine real and fake data
            combined_data = tf.concat([real_data, generated_data], axis=0)

            # Label real data as 1 and fake data as 0
            labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

            # Add random noise to the labels
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            # Compute the discriminator loss
            predictions = self.discriminator(combined_data)
            disc_loss = self.loss_fn(labels, predictions)

        # Compute the gradients of the discriminator loss with respect to the discriminator's weights
        grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        # Update the discriminator's weights
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train Generator
        with tf.GradientTape() as gen_tape:
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            # Generate fake data using the generator
            generated_data = self.generator(random_latent_vectors)

            # Label fake data as 1 (because we want the generator to fool the discriminator)
            misleading_labels = tf.ones((batch_size, 1))

            # Compute the generator loss
            predictions = self.discriminator(generated_data)
            gen_loss = self.loss_fn(misleading_labels, predictions)

        # Compute the gradients of the generator loss with respect to the generator's weights
        grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        # Update the generator's weights
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": disc_loss, "g_loss": gen_loss}

# Loss function
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Instantiate the SeqGAN model
latent_dim = 100
input_dim = 8
seqgan = SeqGAN(latent_dim, input_dim)

# Compile the SeqGAN model
seqgan.compile(gen_optimizer=seqgan.gen_optimizer, disc_optimizer=seqgan.disc_optimizer, loss_fn=loss_fn)

