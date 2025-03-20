import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.EZDiffusion import EZDiffusion

class TestEZDiffusionModel(unittest.TestCase):

    def test_simulate_data(self):
        parameters = EZDiffusion.parameters(self, 1.0, 1.0, 0.3)
        self.assertEqual(len(reaction_times), 50)
        self.assertEqual(len(choices), 50)

    def test_recover_parameters(self):
        reaction_times = np.random.normal(1.0, 0.1, 50)
        choices = np.random.choice([0, 1], size=50)
        recovered = recover_parameters(reaction_times, choices)
        self.assertEqual(len(recovered), 3)

if __name__ == '__main__':
    unittest.main()