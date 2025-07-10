import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 1. 定义 Van der Pol 振荡器动力学方程
def van_der_pol(x, t, mu):
    """Van der Pol 振荡器动力学方程."""
    x1, x2 = x
    dx1dt = x2
    dx2dt = mu * (1 - x1 ** 2) * x2 - x1
    return np.array([dx1dt, dx2dt])  # 确保返回 NumPy 数组


# 2. 生成训练数据
def generate_training_data(mu, T, dt, initial_state):
    """生成 Van der Pol 振荡器训练数据."""
    t = np.arange(0, T, dt)
    solution = odeint(van_der_pol, initial_state, t, args=(mu,))
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    return t, x1, x2


# 3. 定义 DNN 结构
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def d_tanh(x):
    return 1 - tanh(x) ** 2


def d_relu(x):
    return (x > 0).astype(float)


def initialize_dnn(input_size, hidden_sizes, output_size):
    """初始化 DNN 权重."""
    sizes = [input_size] + hidden_sizes + [output_size]
    weights = []
    biases = []
    activations = []
    d_activations = []

    # 根据文档描述，设置激活函数
    activations = [tanh, sigmoid, tanh]
    d_activations = [d_tanh, d_sigmoid, d_tanh]

    for i in range(len(sizes) - 1):
        weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.01)  # 减小初始权重
        biases.append(np.zeros(sizes[i + 1]))

    return weights, biases, activations, d_activations


# 4. 定义 DNN 前向传播
def forward_propagation(x, weights, biases, activations):
    """DNN 前向传播."""
    a = x
    layer_outputs = [x]
    for i in range(len(weights)):
        z = a @ weights[i] + biases[i]
        if i < len(weights) - 1:
            a = activations[i](z)
        else:
            a = z  # 输出层是线性的
        layer_outputs.append(a)
    return layer_outputs


# 5. 定义 DNN 误差函数 (MSE)
def mse_loss(y_true, y_pred):
    """均方误差损失函数."""
    return np.mean((y_true - y_pred) ** 2)


# 6. 定义 DNN 雅可比矩阵 (反向传播)
def backward_propagation(layer_outputs, y_true, weights, biases, activations, d_activations):
    """DNN 反向传播，计算雅可比矩阵."""
    m = len(y_true)
    L = len(weights)
    deltas = [None] * L
    jacobian = []

    # 输出层 delta
    deltas[-1] = (layer_outputs[-1] - y_true)

    # 隐藏层 delta
    for l in range(L - 2, -1, -1):
        deltas[l] = d_activations[l](layer_outputs[l + 1]) * (deltas[l + 1] @ weights[l + 1].T)

    # 计算雅可比矩阵 (简化版本，只考虑最后一层权重)
    for l in range(L):
        jacobian.append(layer_outputs[l].T)

    return deltas, jacobian


# 7. Levenberg-Marquardt 算法 (修改版，用于 DNN 训练)
def levenberg_marquardt_dnn(x_train, y_train, weights, biases, activations, d_activations, max_iterations=100,
                            damping=0.01, tolerance=1e-6, learning_rate=0.001):  # 添加学习率
    """Levenberg-Marquardt 算法训练 DNN."""
    lambda_ = damping
    last_cost = np.inf

    for iteration in range(max_iterations):
        # 前向传播
        layer_outputs = forward_propagation(x_train, weights, biases, activations)

        # 计算残差
        r = y_train - layer_outputs[-1]
        cost = mse_loss(y_train, layer_outputs[-1])

        # 反向传播，计算雅可比矩阵
        deltas, jacobians = backward_propagation(layer_outputs, y_train, weights, biases, activations, d_activations)

        # 计算 Hessian 矩阵的近似和梯度 (简化版本，只更新最后一层权重)
        J = jacobians[-1]
        H = J @ J.T
        g = J @ r

        # 求解增量方程
        try:
            delta = np.linalg.solve(H + lambda_ * np.eye(H.shape[0]), -g)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Increasing damping factor.")
            lambda_ *= 10
            continue

        # 尝试更新参数 (只更新最后一层权重)
        new_weights = weights[-1] - learning_rate * delta.reshape(weights[-1].shape)  # 使用学习率

        # 前向传播，计算新的损失
        new_weights_list = weights[:-1] + [new_weights]
        new_layer_outputs = forward_propagation(x_train, new_weights_list, biases, activations)
        new_cost = mse_loss(y_train, new_layer_outputs[-1])

        # 判断是否接受更新
        if new_cost < cost:
            # 接受更新
            weights[-1] = new_weights
            lambda_ /= 10  # 减小阻尼因子
        else:
            # 拒绝更新
            lambda_ *= 10  # 增大阻尼因子
            new_cost = cost  # 恢复上一次的cost

        # 检查收敛性
        if abs(last_cost - new_cost) < tolerance:
            print(f"Converged after {iteration + 1} iterations. Loss: {new_cost}")
            break

        last_cost = new_cost
        if iteration % 10 == 0:
            print(f"Iteration {iteration + 1}, Loss: {cost}, Damping: {lambda_}")

    return weights


# 8. 控制器设计 (简化版本)
def controller(x, t, dnn_weights, dnn_biases, dnn_activations, k=1.0):
    """简化版控制器."""
    x_d = [5 * np.cos(t), 5 * np.sin(t)]  # 期望轨迹
    e = np.array(x) - np.array(x_d)

    # 使用 DNN 估计 f(x)
    layer_outputs = forward_propagation(np.array(x), dnn_weights, dnn_biases, dnn_activations)
    f_hat = layer_outputs[-1]

    # 简化控制律 (忽略 g(x) 的估计)
    u = -k * e[0] - f_hat[0]  # 只使用一个控制输入, 并且只使用e[0]
    return u


# 9. 仿真
def simulate(mu, dnn_weights, dnn_biases, dnn_activations, T, dt, initial_state):
    """仿真."""
    t = np.arange(0, T, dt)
    x = np.zeros((len(t), 2))
    x[0] = initial_state

    for i in range(len(t) - 1):
        # 计算控制输入
        u = controller(x[i], t[i], dnn_weights, dnn_biases, dnn_activations)

        # 使用控制输入更新状态
        x_dot = van_der_pol(x[i], t[i], mu)
        if not isinstance(x_dot, np.ndarray):
            print(f"Warning: x_dot is not a NumPy array.  Got type {type(x_dot)}")
            break

        x_dot[1] += float(u)  # 将控制输入添加到 x2 的导数上, 确保 u 是标量
        x[i + 1] = x[i] + np.array(x_dot) * dt

    return t, x


# 10. 主程序
if __name__ == "__main__":
    # 仿真参数
    mu = 10.0
    T = 20.0
    dt = 0.01
    initial_state = [-5.0, 8.0]
    desired_trajectory = lambda t: [5 * np.cos(t), 5 * np.sin(t)]

    # DNN 结构
    input_size = 2
    hidden_sizes = [10, 5, 8]
    output_size = 2

    # 1. 生成训练数据
    t_train, x1_train, x2_train = generate_training_data(mu, 600, dt, initial_state)
    x_train = np.column_stack((x1_train, x2_train))
    # 将生成器表达式转换为列表
    y_train = np.column_stack([van_der_pol([x1_train[i], x2_train[i]], t_train[i], mu) for i in range(len(x_train))])
    y_train = np.array(y_train).T

    # 2. 初始化 DNN
    dnn_weights, dnn_biases, dnn_activations, dnn_d_activations = initialize_dnn(input_size, hidden_sizes, output_size)

    # 3. 使用 Levenberg-Marquardt 算法训练 DNN
    trained_weights = levenberg_marquardt_dnn(x_train, y_train, dnn_weights, dnn_biases, dnn_activations,
                                              dnn_d_activations, max_iterations=100, learning_rate=0.001)  # 添加学习率

    # 4. 仿真
    t_sim, x_sim = simulate(mu, trained_weights, dnn_biases, dnn_activations, T, dt, initial_state)

    # 5. 绘制结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t_sim, x_sim[:, 0], label="x1")
    plt.plot(t_sim, x_sim[:, 1], label="x2")
    plt.plot(t_sim, [desired_trajectory(ti)[0] for ti in t_sim], linestyle='--', label="x1_d")
    plt.plot(t_sim, [desired_trajectory(ti)[1] for ti in t_sim], linestyle='--', label="x2_d")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.title("State Trajectories")

    plt.subplot(1, 2, 2)
    plt.plot(x_sim[:, 0], x_sim[:, 1], label="Phase Portrait")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Phase Portrait")

    plt.tight_layout()
    plt.show()
