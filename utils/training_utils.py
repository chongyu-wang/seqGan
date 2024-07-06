# utils/training_utils.py

import tensorflow as tf

def compile_seqgan(seqgan, gen_optimizer, disc_optimizer, loss_fn):
    """
    Compile the SeqGAN model.
    """
    seqgan.compile(gen_optimizer=gen_optimizer, 
                   disc_optimizer=disc_optimizer, 
                   loss_fn=loss_fn)

def train_seqgan(seqgan, data, epochs, batch_size):
    """
    Train the SeqGAN model.
    """
    for epoch in range(epochs):
        for step in range(data.shape[0] // batch_size):
            real_data = data[step * batch_size : (step + 1) * batch_size]
            metrics = seqgan.train_step(real_data)
        
        print(f"Epoch {epoch + 1}/{epochs}, d_loss: {metrics['d_loss']}, g_loss: {metrics['g_loss']}")

def save_checkpoint(model, file_path):
    """
    Save a checkpoint of the model.
    """
    model.save_weights(file_path)

def load_checkpoint(model, file_path):
    """
    Load a checkpoint of the model.
    """
    model.load_weights(file_path)
