import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class VanDerPolSystem:
    """Van der Pol oscillator system dynamics"""

    def __init__(self, mu=10):
        self.mu = mu
        self.g = np.array([[5, 0], [0, 3]])  # Control effectiveness matrix

    def f(self, x):
        """Drift dynamics f(x)"""
        x1, x2 = x[0], x[1]
        return np.array([
            self.mu * (x1 - x1 ** 3 / 3 - x2),
            x1 / self.mu
        ])

    def dynamics(self, t, state, u):
        """System dynamics: dx/dt = f(x) + g(x)u"""
        x = state[:2]
        return self.f(x) + self.g @ u


class DeepNeuralNetwork:
    """Deep Neural Network for function approximation"""

    def __init__(self, input_dim=2, hidden_layers=[10, 5, 8], output_dim=2):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        """Build the DNN architecture"""
        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.Dense(self.hidden_layers[0],
                                     input_dim=self.input_dim,
                                     activation='tanh'))

        # Hidden layers
        model.add(keras.layers.Dense(self.hidden_layers[1],
                                     activation='sigmoid'))
        model.add(keras.layers.Dense(self.hidden_layers[2],
                                     activation='tanh'))

        # Output layer (linear)
        model.add(keras.layers.Dense(self.output_dim,
                                     activation='linear'))

        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mse'])

        return model

    def pretrain(self, X_train, y_train, epochs=535, validation_split=0.15):
        """Pre-train the DNN"""
        # Normalize input data
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train the model
        history = self.model.fit(X_train_scaled, y_train,
                                 epochs=epochs,
                                 validation_split=validation_split,
                                 verbose=0)

        return history

    def predict(self, x):
        """Predict using the trained model"""
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return self.model.predict(x_scaled, verbose=0).flatten()

    def retrain(self, X_new, y_new, epochs=100):
        """Retrain the DNN with new data"""
        X_scaled = self.scaler.transform(X_new)

        # Continue training
        history = self.model.fit(X_scaled, y_new,
                                 epochs=epochs,
                                 verbose=0)

        return history


class AdaptiveController:
    """Lyapunov-based adaptive controller with DNN"""

    def __init__(self, system, dnn, k=75, ks=0.05, gamma_W=500, gamma_theta=np.array([0.1, 0.05])):
        self.system = system
        self.dnn = dnn
        self.k = k
        self.ks = ks
        self.gamma_W = gamma_W * np.eye(13)  # Weight adaptation gain
        self.gamma_theta = np.diag(gamma_theta)  # Parameter adaptation gain

        # Initialize estimates
        self.W_hat = np.random.randn(13, 2) * 0.1  # Output layer weights
        self.theta_hat = np.array([6.0, 6.0])  # Parameter estimates

        # Data storage for retraining
        self.data_x = []
        self.data_f = []

    def desired_trajectory(self, t):
        """Desired trajectory: circular motion"""
        return np.array([5 * np.cos(t), 5 * np.sin(t)])

    def desired_trajectory_dot(self, t):
        """Derivative of desired trajectory"""
        return np.array([-5 * np.sin(t), 5 * np.cos(t)])

    def regression_matrix(self, x, u, t):
        """Regression matrix Y(x,u,t) for parameter estimation"""
        # Simplified regression matrix for this example
        return np.array([[u[0], 0], [0, u[1]]])

    def control_law(self, t, x):
        """Compute control input using adaptive law"""
        # Tracking error
        xd = self.desired_trajectory(t)
        xd_dot = self.desired_trajectory_dot(t)
        e = x - xd

        # DNN approximation
        sigma_hat = self.get_activation_features(x)  # Simplified feature extraction
        f_hat = self.W_hat.T @ sigma_hat

        # Pseudo-inverse of control effectiveness matrix
        g_inv = np.linalg.pinv(self.system.g)

        # Control law
        u = g_inv @ (-self.k * e - self.ks * np.sign(e) + xd_dot - f_hat)

        return u, e, sigma_hat, f_hat

    def get_activation_features(self, x):
        """Extract activation features from DNN (simplified)"""
        # This is a simplified version - in practice, you'd extract features
        # from the intermediate layers of the DNN
        features = np.zeros(13)
        features[0] = np.tanh(x[0])
        features[1] = np.tanh(x[1])
        features[2] = 1.0 / (1.0 + np.exp(-x[0]))  # sigmoid
        features[3] = 1.0 / (1.0 + np.exp(-x[1]))
        features[4] = np.tanh(x[0] + x[1])
        features[5] = np.tanh(x[0] - x[1])
        features[6] = x[0]
        features[7] = x[1]
        features[8] = x[0] ** 2
        features[9] = x[1] ** 2
        features[10] = x[0] * x[1]
        features[11] = np.sin(x[0])
        features[12] = np.cos(x[1])
        return features

    def update_weights(self, e, sigma_hat):
        """Update output layer weights using Lyapunov-based adaptation"""
        self.W_hat += self.gamma_W @ sigma_hat.reshape(-1, 1) @ e.reshape(1, -1)

    def update_parameters(self, x, u, t, e):
        """Update parameter estimates"""
        Y = self.regression_matrix(x, u, t)
        self.theta_hat += self.gamma_theta @ Y.T @ e

    def collect_data(self, x, f_true):
        """Collect data for DNN retraining"""
        self.data_x.append(x.copy())
        self.data_f.append(f_true.copy())


def generate_training_data(system, duration=600, dt=0.01):
    """Generate training data for DNN pre-training"""
    t_span = np.arange(0, duration, dt)
    n_samples = len(t_span)

    X_train = []
    y_train = []

    # Generate random initial conditions and collect data
    for _ in range(100):  # Multiple trajectories
        x0 = np.random.uniform(-5, 5, 2)

        for i in range(min(100, n_samples)):  # Limit samples per trajectory
            x = x0 + 0.1 * np.random.randn(2)  # Add some noise
            f_x = system.f(x)

            X_train.append(x)
            y_train.append(f_x)

    return np.array(X_train), np.array(y_train)


def simulate_adaptive_control():
    """Main simulation function"""
    # System parameters
    system = VanDerPolSystem(mu=10)

    # Create and pre-train DNN
    dnn = DeepNeuralNetwork()

    # Generate pre-training data
    print("Generating pre-training data...")
    X_train, y_train = generate_training_data(system)

    # Pre-train DNN
    print("Pre-training DNN...")
    history = dnn.pretrain(X_train, y_train)

    # Create adaptive controller
    controller = AdaptiveController(system, dnn)

    # Simulation parameters
    t_final = 100.0
    dt = 0.01
    t_span = np.arange(0, t_final, dt)

    # Initial conditions
    x0 = np.array([-5.0, 8.0])

    # Storage for results
    x_history = []
    u_history = []
    e_history = []
    xd_history = []
    W_norm_history = []

    # Retraining parameters
    retrain_interval = 25.0  # seconds
    retrain_times = [25.0, 62.4]
    retrain_completed = [37.4, 68.3]
    current_retrain_idx = 0

    print("Starting simulation...")

    # Simulation loop
    x = x0.copy()
    def system_dynamics(t, x):
        u, e, sigma_hat, f_hat = controller.control_law(t, x[:2])
        controller.collect_data(x[:2], system.f(x[:2]))
        controller.update_weights(e, sigma_hat)
        controller.update_parameters(x[:2], u, t, e)
        return system.dynamics(t, x[:2], u)

    sol = solve_ivp(system_dynamics, [0, t_final], x0, dense_output=True, max_step=dt)

    t_span = sol.t
    x_history = sol.y.T

    # Compute other data based on the solution
    u_history = []
    e_history = []
    xd_history = []
    W_norm_history = []

    for i, t in enumerate(t_span):
        x = x_history[i]
        u, e, sigma_hat, f_hat = controller.control_law(t, x)
        xd = controller.desired_trajectory(t)

        u_history.append(u.copy())
        e_history.append(e.copy())
        xd_history.append(xd.copy())
        W_norm_history.append(np.linalg.norm(controller.W_hat))

    u_history = np.array(u_history)
    e_history = np.array(e_history)
    xd_history = np.array(xd_history)
    W_norm_history = np.array(W_norm_history)

    return (t_span, x_history, u_history, e_history, xd_history,
            W_norm_history, retrain_times, retrain_completed)


def plot_results(results):
    """Plot simulation results"""
    (t_span, x_history, u_history, e_history, xd_history,
     W_norm_history, retrain_times, retrain_completed) = results

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Deep Neural Network Adaptive Control Simulation Results', fontsize=16)

    # Tracking error
    axes[0, 0].plot(t_span, np.linalg.norm(e_history, axis=1), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Tracking Error ||e||')
    axes[0, 0].set_title('Tracking Error Over Time')
    axes[0, 0].grid(True)

    # Add retraining markers
    for i, (start, end) in enumerate(zip(retrain_times, retrain_completed)):
        axes[0, 0].axvline(x=start, color='r', linestyle='--', alpha=0.7,
                           label=f'Retrain {i + 1} Start' if i == 0 else '')
        axes[0, 0].axvline(x=end, color='k', linestyle='--', alpha=0.7, label=f'Retrain {i + 1} End' if i == 0 else '')

    if len(retrain_times) > 0:
        axes[0, 0].legend()

    # State trajectories
    axes[0, 1].plot(t_span, x_history[:, 0], 'b-', label='x₁', linewidth=2)
    axes[0, 1].plot(t_span, x_history[:, 1], 'r-', label='x₂', linewidth=2)
    axes[0, 1].plot(t_span, xd_history[:, 0], 'b--', label='x₁ᵈ', alpha=0.7)
    axes[0, 1].plot(t_span, xd_history[:, 1], 'r--', label='x₂ᵈ', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('State')
    axes[0, 1].set_title('State Trajectories')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Phase plot
    axes[0, 2].plot(x_history[:, 0], x_history[:, 1], 'b-', linewidth=2, label='Actual')
    axes[0, 2].plot(xd_history[:, 0], xd_history[:, 1], 'r--', linewidth=2, label='Desired')
    axes[0, 2].set_xlabel('x₁')
    axes[0, 2].set_ylabel('x₂')
    axes[0, 2].set_title('Phase Plot')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].axis('equal')

    # Control inputs
    axes[1, 0].plot(t_span, u_history[:, 0], 'b-', label='u₁', linewidth=2)
    axes[1, 0].plot(t_span, u_history[:, 1], 'r-', label='u₂', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Input')
    axes[1, 0].set_title('Control Inputs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Weight norm evolution
    axes[1, 1].plot(t_span, W_norm_history, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('||Ŵ||')
    axes[1, 1].set_title('Weight Estimates Norm')
    axes[1, 1].grid(True)

    # Individual tracking errors
    axes[1, 2].plot(t_span, e_history[:, 0], 'b-', label='e₁', linewidth=2)
    axes[1, 2].plot(t_span, e_history[:, 1], 'r-', label='e₂', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Tracking Error')
    axes[1, 2].set_title('Individual Tracking Errors')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate and print RMSE
    rmse_total = np.sqrt(np.mean(np.sum(e_history ** 2, axis=1)))
    rmse_e1 = np.sqrt(np.mean(e_history[:, 0] ** 2))
    rmse_e2 = np.sqrt(np.mean(e_history[:, 1] ** 2))

    print(f"\n=== Performance Metrics ===")
    print(f"Total RMSE: {rmse_total:.6f}")
    print(f"RMSE e₁: {rmse_e1:.6f}")
    print(f"RMSE e₂: {rmse_e2:.6f}")
    print(f"Final tracking error: {np.linalg.norm(e_history[-1]):.6f}")


if __name__ == "__main__":
    # Run simulation
    print("Starting Deep Neural Network Adaptive Control Simulation")
    print("=" * 60)

    # Run the simulation
    results = simulate_adaptive_control()

    # Plot results
    plot_results(results)

    print("\nSimulation completed successfully!")
