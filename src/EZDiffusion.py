import numpy as np

class EZDiffusion:
    def forward_equations(v, a, t):
        y = np.exp(-v * a)
        R_pred = 1 / (1 + y)
        M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
        return R_pred, M_pred, V_pred

    def simulate_observed_statistics(R_pred, M_pred, V_pred, N):
        """Generate observed summary statistics using sampling distributions."""
        T_obs = np.random.binomial(N, R_pred)
        #R_obs = T_obs / N
        R_obs = max(min(T_obs / N, 0.999), 0.001) #Used AI for this line, because I kept getting an error about dividing by zero
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
        V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))
        return R_obs, M_obs, V_obs

    def inverse_equations(R_obs, M_obs, V_obs):
        """Recover EZ diffusion parameters from observed statistics."""
        L = np.log(R_obs / (1 - R_obs))
        v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)
        a_est = L / v_est
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
        return v_est, a_est, t_est

    def simulate_and_recover(N, iterations=1000):
        """Perform simulate-and-recover for given N and iterations."""
        biases = []
        squared_errors = []
        
        for _ in range(iterations):
            # Sample true parameters
            v_true = np.random.uniform(0.5, 2)
            a_true = np.random.uniform(0.5, 2)
            t_true = np.random.uniform(0.1, 0.5)
            
            # Forward equations
            R_pred, M_pred, V_pred = EZDiffusion.forward_equations(v_true, a_true, t_true)
            
            # Simulate observed data
            R_obs, M_obs, V_obs = EZDiffusion.simulate_observed_statistics(R_pred, M_pred, V_pred, N)
            
            # Recover parameters
            v_est, a_est, t_est = EZDiffusion.inverse_equations(R_obs, M_obs, V_obs)
            
            # Compute bias and squared error
            bias = np.array([v_true - v_est, a_true - a_est, t_true - t_est])
            biases.append(bias)
            squared_errors.append(bias**2)
        
        biases = np.array(biases).mean(axis=0)
        squared_errors = np.array(squared_errors).mean(axis=0)
        
        return biases, squared_errors

def main():
    """Run the full simulate-and-recover study for N = 10, 40, 4000."""
    for N in [10, 40, 4000]:
        biases, squared_errors = EZDiffusion.simulate_and_recover(N)
        print(f"N={N}: Bias={biases}, Squared Error={squared_errors}")

if __name__ == "__main__":
    main()
