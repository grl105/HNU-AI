import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    """加载数据"""
    data = np.loadtxt(path, delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]
    return X, y


def feature_normalize(X):
    """特征标准化"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def plot_data(X, y, title=''):
    """绘制数据点"""
    plt.figure(figsize=(10, 6))
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='green', marker='+',
                label='Admitted', s=100)
    plt.scatter(X[neg, 0], X[neg, 1], c='red', marker='o',
                label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend()
    if title:
        plt.title(title)
    plt.grid(True)


def sigmoid(z):
    """Sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(X, y, theta):
    """计算代价函数值和梯度"""
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return J, grad


def gradient_descent(X, y, learning_rate=0.01, num_iters=1000):
    """梯度下降优化"""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []  # 记录cost历史

    initial_cost, _ = compute_cost(X, y, theta)
    print(f"初始代价函数值: {initial_cost:.4f}")
    cost_history.append(initial_cost)

    print("开始训练模型:")
    for i in range(num_iters):
        cost, grad = compute_cost(X, y, theta)
        theta = theta - learning_rate * grad
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, cost_history


def plot_cost_history(cost_history):
    """绘制代价函数的收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.grid(True)
    plt.show()


def plot_decision_boundary(theta, X, y, mu, sigma):
    """绘制决策边界"""
    plot_data(X, y, 'Decision Boundary')

    # 计算决策边界点
    x1_min, x1_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    x2_min, x2_max = X[:, 1].min() - 2, X[:, 1].max() + 2

    # 生成网格点
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))

    # 将网格点标准化
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_points_norm = (grid_points - mu) / sigma

    # 添加截距项并计算预测值
    grid_points_norm = np.c_[np.ones(grid_points_norm.shape[0]), grid_points_norm]
    predictions = sigmoid(np.dot(grid_points_norm, theta))

    # 重塑预测值并绘制等高线
    predictions = predictions.reshape(xx1.shape)
    plt.contour(xx1, xx2, predictions, levels=[0.5], colors='blue', label='Decision Boundary')

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.legend()
    plt.show()


def predict(X, theta):
    """预测函数"""
    return sigmoid(np.dot(X, theta))


# 主程序
if __name__ == "__main__":
    # 1. 加载数据
    X, y = load_data('ex2data1.txt')
    print("数据分布可视化：")
    plot_data(X, y, 'Admission Data')
    plt.show()

    # 2. 特征标准化
    X_norm, mu, sigma = feature_normalize(X)

    # 3. 添加截距项
    X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

    # 4. 训练模型
    theta, cost_history = gradient_descent(X_b, y, learning_rate=0.1, num_iters=5000)

    # 4.1 绘制代价函数收敛曲线
    print("\n代价函数收敛过程：")
    plot_cost_history(cost_history)

    # 5. 预测新学生
    x_test = np.array([45, 85])
    x_test_norm = (x_test - mu) / sigma
    x_test_b = np.r_[1, x_test_norm]
    prob = predict(x_test_b, theta)
    print(f"\n考试成绩为[45, 85]的学生被录取的概率为: {prob * 100:.2f}%")

    # 6. 绘制决策边界
    print("决策边界可视化：")
    plot_decision_boundary(theta, X, y, mu, sigma)

    # 7. 打印模型准确率
    predictions = predict(X_b, theta) >= 0.5
    accuracy = np.mean(predictions == y)
    print(f"\n模型准确率: {accuracy * 100:.2f}%")