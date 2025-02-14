import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore import context
import matplotlib.pyplot as plt

# 设置MindSpore运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def load_data(filename):
    """加载数据并返回特征矩阵X和目标值y"""
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:2]  # 前两列为特征
    y = data[:, 2]  # 第三列为房价
    return X, y


def feature_normalize(X):
    """特征归一化处理"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


class LinearRegression(nn.Cell):
    """线性回归模型类"""

    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.weights = ms.Parameter(ms.Tensor(np.zeros((input_dim, 1)), ms.float32))
        self.bias = ms.Parameter(ms.Tensor(np.zeros(1), ms.float32))

    def construct(self, x):
        return ops.matmul(x, self.weights) + self.bias


def compute_cost(model, X, y):
    """计算损失函数"""
    m = len(y)
    predictions = model(X)
    cost = ops.reduce_mean(ops.square(predictions - y.reshape(-1, 1))) / 2.0
    return cost


def gradient_descent(model, X, y, alpha, num_iters):
    """梯度下降优化"""
    m = len(y)
    costs = []

    optimizer = nn.SGD(model.trainable_params(), learning_rate=alpha)

    def forward_fn(X, y):
        cost = compute_cost(model, X, y)
        return cost

    grad_fn = ops.value_and_grad(forward_fn, None, model.trainable_params())

    for i in range(num_iters):
        cost, grads = grad_fn(X, y)
        optimizer(grads)
        costs.append(float(cost))

        if i % 100 == 0:
            print(f'Iteration {i}: Cost = {float(cost)}')

    return costs


def main():
    # 1. 加载数据
    X, y = load_data('ex1data2.txt')

    # 2. 特征归一化
    X_norm, mu, sigma = feature_normalize(X)

    # 将数据转换为MindSpore张量
    X_ms = ms.Tensor(X_norm, ms.float32)
    y_ms = ms.Tensor(y, ms.float32)

    # 3. 创建和训练模型
    model = LinearRegression(input_dim=2)

    # 尝试不同的学习率
    alphas = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    plt.figure(figsize=(12, 8))

    for alpha in alphas:
        print(f"\n训练模型 (alpha = {alpha}):")
        model = LinearRegression(input_dim=2)  # 为每个学习率创建新的模型
        costs = gradient_descent(model, X_ms, y_ms, alpha, num_iters=400)
        plt.plot(range(len(costs)), costs, label=f'alpha = {alpha}')

    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. 使用最佳学习率重新训练模型
    best_alpha = 0.1  # 手动设置最佳学习率
    print("\n使用最佳学习率重新训练模型:")
    model = LinearRegression(input_dim=2)
    costs = gradient_descent(model, X_ms, y_ms, best_alpha, num_iters=400)

    # 5. 预测房价
    # 对新数据进行归一化
    test_X = np.array([[1650, 3]])
    test_X_norm = (test_X - mu) / sigma
    test_X_ms = ms.Tensor(test_X_norm, ms.float32)

    # 进行预测
    prediction = model(test_X_ms)
    predicted_price = float(prediction.asnumpy()[0][0])
    print(f"\n预测结果:")
    print(f"面积为1650平方英尺，3个房间的房屋预测价格为: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()