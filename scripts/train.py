# scripts/train.py

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seqgan import SeqGAN
from models.generator import build_generator
from models.discriminator import build_discriminator

def load_processed_data(file_path):
    """
    Load the processed data from a CSV file.
    """
    return pd.read_csv(file_path).values

def train_seqgan(epochs, batch_size, latent_dim, data):
    """
    Train the SeqGAN model.
    """
    input_dim = data.shape[1]
    
    # Initialize the SeqGAN model
    seqgan = SeqGAN(latent_dim, input_dim)
    
    # Compile the SeqGAN model
    seqgan.compile(gen_optimizer=seqgan.gen_optimizer, 
                   disc_optimizer=seqgan.disc_optimizer, 
                   loss_fn=tf.keras.losses.BinaryCrossentropy())
    
    for epoch in range(epochs):
        for step in range(data.shape[0] // batch_size):
            real_data = data[step * batch_size : (step + 1) * batch_size]
            metrics = seqgan.train_step(real_data)
        
        print(f"Epoch {epoch + 1}/{epochs}, d_loss: {metrics['d_loss']}, g_loss: {metrics['g_loss']}")
        
    return seqgan

if __name__ == "__main__":
    # Configuration
    epochs = 100
    batch_size = 64
    latent_dim = 100
    processed_data_path = os.path.join("data", "processed", "patient_vital_signs_normalized.csv")
    
    # Load processed data
    data = load_processed_data(processed_data_path)
    
    # Train SeqGAN
    trained_seqgan = train_seqgan(epochs, batch_size, latent_dim, data)
    
    # Save the trained generator model
    generator_path = os.path.join("models", "trained_generator.keras")
    trained_seqgan.generator.save(generator_path)
    print(f"Trained generator model saved to {generator_path}")




