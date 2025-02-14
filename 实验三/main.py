import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt


# 1. 数据加载与可视化
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0].reshape(-1, 1)  # 人口数据
    y = data[:, 1].reshape(-1, 1)  # 利润数据
    return X, y


def visualize_data(X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Training Data')
    plt.title('Population vs Profit')
    plt.xlabel('Population (10,000)')
    plt.ylabel('Profit ($10,000)')
    plt.legend()
    plt.show()


# 2. 线性回归模型
class LinearRegressionModel(nn.Cell):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        # 初始化权重和偏置，均设为0
        self.weights = mindspore.Parameter(
            mindspore.Tensor(np.zeros((input_dim, 1)), mindspore.float32),
            name='weights'
        )
        self.bias = mindspore.Parameter(
            mindspore.Tensor(np.zeros((1, 1)), mindspore.float32),
            name='bias'
        )

    def construct(self, x):
        # 线性回归预测函数 y = wx + b
        return ops.matmul(x, self.weights) + self.bias


# 3. 代价函数（均方误差）
def compute_cost(model, X, y):
    m = X.shape[0]
    predictions = model(X)
    cost = ops.reduce_mean((predictions - y) ** 2) / (2 * m)
    return cost


# 4. 梯度下降训练
def gradient_descent(X, y, model, learning_rate, iterations):
    m = X.shape[0]
    costs = []

    # 转换为Tensor
    X_tensor = mindspore.Tensor(X, mindspore.float32)
    y_tensor = mindspore.Tensor(y, mindspore.float32)

    for i in range(iterations):
        # 前向传播
        predictions = model(X_tensor)

        # 计算梯度
        dw = ops.matmul(X_tensor.T, (predictions - y_tensor)) / m
        db = ops.reduce_sum(predictions - y_tensor) / m

        # 更新参数
        model.weights -= learning_rate * dw
        model.bias -= learning_rate * db

        # 记录代价
        cost = compute_cost(model, X_tensor, y_tensor)
        costs.append(cost.asnumpy())

        # 每100次迭代打印一次代价
        if i % 100 == 0:
            print(f'Iteration {i}: Cost = {cost.asnumpy()}')

    return costs


# 5. 可视化拟合结果
def plot_regression_line(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Training Data')

    # 绘制回归线
    X_sorted = np.sort(X, axis=0)
    y_pred = model(mindspore.Tensor(X_sorted, mindspore.float32)).asnumpy()

    plt.plot(X_sorted, y_pred, color='blue', label='Regression Line')
    plt.title('Population vs Profit - Regression Line')
    plt.xlabel('Population (10,000)')
    plt.ylabel('Profit ($10,000)')
    plt.legend()
    plt.show()


# 主函数
def main():
    # 加载数据
    X, y = load_data('ex1data1.txt')

    # 1. 数据可视化
    visualize_data(X, y)

    # 2. 初始化模型
    model = LinearRegressionModel(input_dim=1)

    # 3. 初始代价
    X_tensor = mindspore.Tensor(X, mindspore.float32)
    y_tensor = mindspore.Tensor(y, mindspore.float32)
    initial_cost = compute_cost(model, X_tensor, y_tensor)
    print(f'Initial Cost: {initial_cost.asnumpy()}')

    # 4. 梯度下降训练
    learning_rate = 0.01
    iterations = 1500
    costs = gradient_descent(X, y, model, learning_rate, iterations)

    # 5. 可视化拟合结果
    plot_regression_line(X, y, model)

    # 6. 预测
    def predict(population):
        pop_tensor = mindspore.Tensor([[population]], mindspore.float32)
        prediction = model(pop_tensor).asnumpy()
        return prediction[0][0]

    print(f'Prediction for 35000 population: ${predict(3.5):.2f} thousand')
    print(f'Prediction for 70000 population: ${predict(7.0):.2f} thousand')

    # 打印最终参数
    print(f'Final Weights: {model.weights.asnumpy()}')
    print(f'Final Bias: {model.bias.asnumpy()}')


if __name__ == "__main__":
    main()