import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

# 设置MindSpore运行模式
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class BPNeuralNetwork(nn.Cell):
    """BP神经网络模型"""

    def __init__(self):
        super(BPNeuralNetwork, self).__init__()
        # 使用三层神经网络结构
        self.layer1 = nn.Dense(3, 16)
        self.layer2 = nn.Dense(16, 8)
        self.layer3 = nn.Dense(8, 1)
        # 使用ReLU激活函数
        self.relu = nn.ReLU()
        # 添加BatchNorm层
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

    def construct(self, x):
        x = self.bn1(self.relu(self.layer1(x)))
        x = self.bn2(self.relu(self.layer2(x)))
        x = self.layer3(x)
        return x


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)


class TrafficPredictor:
    def __init__(self):
        try:
            # 读取Excel文件中的历史数据 (list1)
            df = pd.read_excel('list1.xls')

            # 假设Excel文件的列顺序为：年份,人口数量,机动车数量,公路面积,客运量,货运量
            self.years = df.iloc[:, 0].values  # 第一列是年份
            self.population = df.iloc[:, 1].values  # 人口数量
            self.vehicles = df.iloc[:, 2].values  # 机动车数量
            self.road_area = df.iloc[:, 3].values  # 公路面积
            self.passengers = df.iloc[:, 4].values  # 客运量
            self.freight = df.iloc[:, 5].values  # 货运量

            # 2010和2011年的预测数据
            self.predict_population = np.array([73.39, 75.55])  # 人口数量
            self.predict_vehicles = np.array([3.9635, 4.0975])  # 机动车数量
            self.predict_road_area = np.array([0.9880, 1.0268])  # 公路面积

            # 初始化标准化器
            self.scaler_X = MinMaxScaler()
            self.scaler_y1 = MinMaxScaler()
            self.scaler_y2 = MinMaxScaler()

            # 准备训练数据
            self.X = np.column_stack((self.population, self.vehicles, self.road_area))
            self.X_scaled = self.scaler_X.fit_transform(self.X)
            self.y1_scaled = self.scaler_y1.fit_transform(self.passengers.reshape(-1, 1))
            self.y2_scaled = self.scaler_y2.fit_transform(self.freight.reshape(-1, 1))

            # 准备预测数据
            self.X_pred = np.column_stack((self.predict_population, self.predict_vehicles, self.predict_road_area))
            self.X_pred_scaled = self.scaler_X.transform(self.X_pred)

            # 转换为MindSpore张量
            self.X_scaled_ms = Tensor(self.X_scaled.astype(np.float32))
            self.y1_scaled_ms = Tensor(self.y1_scaled.astype(np.float32))
            self.y2_scaled_ms = Tensor(self.y2_scaled.astype(np.float32))
            self.X_pred_scaled_ms = Tensor(self.X_pred_scaled.astype(np.float32))

        except Exception as e:
            print(f"读取数据时发生错误: {str(e)}")
            raise

    def train_single_model(self, model, X, y, learning_rate=0.001, epochs=2000):
        """训练单个模型"""
        loss_fn = nn.MSELoss()
        net_with_loss = NetWithLoss(model, loss_fn)
        optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

        def forward_fn(data, target):
            return net_with_loss(data, target)

        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

        for epoch in range(epochs):
            loss, grads = grad_fn(X, y)
            optimizer(grads)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.asnumpy():.6f}")

        return model

    def train_models(self):
        """训练BP神经网络和线性回归模型"""
        # 训练BP神经网络模型
        print("\n训练BP神经网络模型...")

        # 客运量预测模型
        self.bp_model_passengers = BPNeuralNetwork()
        print("训练客运量预测模型...")
        self.bp_model_passengers = self.train_single_model(
            self.bp_model_passengers,
            self.X_scaled_ms,
            self.y1_scaled_ms
        )

        # 货运量预测模型
        self.bp_model_freight = BPNeuralNetwork()
        print("\n训练货运量预测模型...")
        self.bp_model_freight = self.train_single_model(
            self.bp_model_freight,
            self.X_scaled_ms,
            self.y2_scaled_ms
        )

        # BP预测结果
        self.bp_pred_passengers = self.scaler_y1.inverse_transform(
            self.bp_model_passengers(self.X_pred_scaled_ms).asnumpy().reshape(-1, 1))
        self.bp_pred_freight = self.scaler_y2.inverse_transform(
            self.bp_model_freight(self.X_pred_scaled_ms).asnumpy().reshape(-1, 1))

        # 训练线性回归模型
        print("\n训练线性回归模型...")
        self.lr_passengers = LinearRegression()
        self.lr_freight = LinearRegression()

        self.lr_passengers.fit(self.X_scaled, self.y1_scaled)
        self.lr_freight.fit(self.X_scaled, self.y2_scaled)

        # 线性回归预测结果
        self.lr_pred_passengers = self.scaler_y1.inverse_transform(
            self.lr_passengers.predict(self.X_pred_scaled).reshape(-1, 1))
        self.lr_pred_freight = self.scaler_y2.inverse_transform(
            self.lr_freight.predict(self.X_pred_scaled).reshape(-1, 1))

    def print_predictions(self):
        """打印预测结果"""
        print("\nBP Neural Network Predictions:")
        print("2010 Passenger Traffic: {:.0f} (10k people)".format(self.bp_pred_passengers[0][0]))
        print("2011 Passenger Traffic: {:.0f} (10k people)".format(self.bp_pred_passengers[1][0]))
        print("2010 Freight Traffic: {:.0f} (10k tons)".format(self.bp_pred_freight[0][0]))
        print("2011 Freight Traffic: {:.0f} (10k tons)".format(self.bp_pred_freight[1][0]))

        print("\nLinear Regression Predictions:")
        print("2010 Passenger Traffic: {:.0f} (10k people)".format(self.lr_pred_passengers[0][0]))
        print("2011 Passenger Traffic: {:.0f} (10k people)".format(self.lr_pred_passengers[1][0]))
        print("2010 Freight Traffic: {:.0f} (10k tons)".format(self.lr_pred_freight[0][0]))
        print("2011 Freight Traffic: {:.0f} (10k tons)".format(self.lr_pred_freight[1][0]))

    def plot_results(self):
        """Visualization of results"""
        # Set font to DejaVu Sans explicitly
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.figure(figsize=(15, 6))

        # Passenger traffic plot
        plt.subplot(1, 2, 1)
        plt.plot(self.years, self.passengers, 'bo-', label='Historical Passenger Traffic')
        plt.plot([2010, 2011], self.bp_pred_passengers, 'ro-', label='BP Prediction')
        plt.plot([2010, 2011], self.lr_pred_passengers, 'go-', label='Linear Regression')
        plt.xlabel('Year')
        plt.ylabel('Passenger Traffic (10k people)')
        plt.legend()
        plt.grid(True)

        # Freight traffic plot
        plt.subplot(1, 2, 2)
        plt.plot(self.years, self.freight, 'bo-', label='Historical Freight Traffic')
        plt.plot([2010, 2011], self.bp_pred_freight, 'ro-', label='BP Prediction')
        plt.plot([2010, 2011], self.lr_pred_freight, 'go-', label='Linear Regression')
        plt.xlabel('Year')
        plt.ylabel('Freight Traffic (10k tons)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    try:
        predictor = TrafficPredictor()
        predictor.train_models()
        predictor.print_predictions()
        predictor.plot_results()
    except Exception as e:
        print(f"错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()