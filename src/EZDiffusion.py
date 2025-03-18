import numpy as np
import scipy.stats as stats

class EZDiffusion:

    def __init__(self, drift, boundary, nondecision, N):
        self.drift = drift
        self.boundary = boundary
        self.nondecision = nondecision
        self.N = N

    def forward(self, r_pred, m_pred, v_pred):
        y = np.exp(-self.boundary * self.drift)
        self.r_pred = r_pred
        r_pred = 1 / (1 + y) #EQ 1
        self.m_pred = m_pred
        m_pred = self.nondecision + (self.boundary / (2 * self.drift)) * ((1 - y) / (1 + y)) #EQ 2
        self.v_pred = v_pred
        v_pred = (self.boundary / (2 * self.drift**3)) * ((1 - 2 * self.boundary * self.drift * y - y**2) / (1 + y)**2) #EQ 3

    def distributions(self, N):
        r_pred, m_pred, v_pred = self.forward() #What does this line do?
        ##T_obs.self = T_obs
        T_obs = np.random.binomial(N, r_pred) #EQ 7
        ##R_obs.self = R_obs
        R_obs = T_obs / N #EQ 7
        ##M_obs.self = M_obs
        M_obs = np.random.normal(m_pred, np.sqrt(v_pred / N)) #EQ 8
        ##V_obs.self = V_obs
        V_obs = np.random.gamma((N - 1) / 2, (2 * v_pred) / (N - 1)) #EQ 9 CHECK SHAPE AND SCALE???
    
    def inverse(R_obs, M_obs, V_obs):
        L = np.log(R_obs / (1 - R_obs))
        v_est = np.sign(R_obs - 0.5) * np.sqrt(L / (V_obs * ((L**2 / R_obs**2) * (L - R_obs) / (L + R_obs) - 0.5))) #EQ 4
        a_est = L / v_est #EQ 5
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))



