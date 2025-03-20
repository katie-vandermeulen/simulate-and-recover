import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.EZDiffusion import EZDiffusion

class Test_EZDiffusion(unittest.TestCase):
    def setUp(self):
        self.simulator = EZDiffusion(iterations=500)

    def test_forward(self):
        v, a, t = 1.0, 1.5, 2.0
        r_pred, m_pred, v_pred = self.simulator.forward(v, a, t)
        self.assertTrue(0 <= r_pred <= 1)
        self.assertTrue(m_pred > t)
        self.assertTrue(v_pred > 0)
    
    def test_distributions(self):
        v, a, t = 1.0, 1.5, 2.0
        r_pred, m_pred, v_pred = self.simulator.forward(v, a, t)
        N = 1000
        T_obs, R_obs, V_obs, M_obs = self.simulator.distributions(N, r_pred, m_pred, v_pred)

        self.assertIsInstance(T_obs, int)
        self.assertTrue(0 <= R_obs <= 1)
        self.assertTrue(V_obs > 0)
        self.assertTrue(M_obs > 0)

    def test_inverse(self):
        v, a, t = 1.0, 1.5, 2.0
        r_pred, m_pred, v_pred = self.simulator.forward(v, a, t)
        N = 1000

        T_obs, R_obs, V_obs, M_obs = self.simulator.distributions(N, r_pred, m_pred, v_pred)
        v_est, a_est, t_est = self.simulator.inverse(R_obs, V_obs, M_obs)

        self.assertAlmostEqual(v, v_est, delta=0.2)
        self.assertAlmostEqual(a, a_est, delta=0.2)
        self.assertAlmostEqual(t, t_est, delta=0.1)

if __name__ == "__main__":
    unittest.main()