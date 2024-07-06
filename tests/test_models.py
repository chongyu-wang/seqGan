# tests/test_models.py

import unittest
import numpy as np
from models.generator import build_generator
from models.discriminator import build_discriminator

class TestModels(unittest.TestCase):

    def setUp(self):
        self.latent_dim = 100
        self.input_dim = 30
        self.generator = build_generator(self.latent_dim)
        self.discriminator = build_discriminator(self.input_dim)

    def test_generator_output_shape(self):
        random_input = np.random.normal(size=(1, self.latent_dim))
        generated_data = self.generator(random_input)
        self.assertEqual(generated_data.shape, (1, 30))

    def test_discriminator_output_shape(self):
        real_input = np.random.normal(size=(1, self.input_dim))
        decision = self.discriminator(real_input)
        self.assertEqual(decision.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()

