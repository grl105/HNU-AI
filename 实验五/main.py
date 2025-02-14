import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.dataset.transforms import TypeCast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


# 自定义SVM分类器
class CustomSVM(nn.Cell):
    def __init__(self, input_dim, num_classes):
        super(CustomSVM, self).__init__()
        # 增加隐藏层，使模型能够学习更复杂的特征
        self.fc1 = nn.Dense(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Dense(64, num_classes)

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_iris_data(filename):
    # 使用空格分隔符读取数据，跳过第一行
    data = []
    with open(filename, 'r') as file:
        # 跳过标题行
        next(file)
        for line in file:
            # 移除开头的引号和结尾的换行符
            if line.startswith('"'):
                line = line[1:]
            # 分割行并处理每个字段
            row = line.strip().split(' ')
            # 提取数值和类别
            values = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            label = row[5].strip('"')
            data.append(values + [label])

    # 转换为numpy数组
    data_array = np.array(data)
    X = data_array[:, :4].astype(np.float32)
    y = pd.Categorical(data_array[:, 4]).codes.astype(np.int32)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data shape:", X.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Sample of features:", X[:5])
    print("Sample of labels:", y[:5])

    return X_scaled, y


def create_dataset(features, labels, batch_size=32, shuffle=True):
    data = {'features': features, 'labels': labels}
    dataset = ds.NumpySlicesDataset(data, shuffle=shuffle)

    type_cast_op = TypeCast(ms.float32)
    dataset = dataset.map(operations=type_cast_op, input_columns=['features'])
    dataset = dataset.map(operations=TypeCast(ms.int32), input_columns=['labels'])
    dataset = dataset.batch(batch_size)

    return dataset


def train_model(dataset, input_dim, num_classes=3, epochs=200):  # 增加训练轮数
    # 创建模型和损失函数
    model = CustomSVM(input_dim, num_classes)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 使用更小的学习率
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01, weight_decay=0.01)  # 添加L2正则化

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # 训练循环
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for data in dataset.create_dict_iterator():
            features = data['features']
            labels = data['labels']
            loss = train_step(features, labels)
            total_loss += float(loss)
            steps += 1

        if (epoch + 1) % 20 == 0:  # 每20轮打印一次
            print(f'Epoch {epoch + 1}, Average Loss: {total_loss / steps}')

    return model


def cross_validation(X, y, k_folds=10):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k_folds

    accuracies = []
    for i in range(k_folds):
        print(f"Processing fold {i + 1}/{k_folds}")
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size],
                                        indices[(i + 1) * fold_size:]])

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        train_dataset = create_dataset(X_train, y_train)
        model = train_model(train_dataset, X.shape[1])

        # 评估
        val_logits = model(ms.Tensor(X_val))
        val_preds = ops.argmax(val_logits, 1)
        accuracy = (val_preds.asnumpy() == y_val).mean()
        accuracies.append(accuracy)
        print(f"Fold {i + 1} accuracy: {accuracy:.4f}")

    return np.array(accuracies)


def pca_visualization(X, y):
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 训练模型
    dataset = create_dataset(X_pca, y)
    model = train_model(dataset, input_dim=2)

    # 创建网格
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_tensor = ms.Tensor(grid_points)
    logits = model(grid_tensor)
    Z = ops.argmax(logits, 1).asnumpy()
    Z = Z.reshape(xx.shape)

    # 可视化
    plt.figure(figsize=(10, 8))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolors='black')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('SVM Decision Boundary after PCA (MindSpore)')

    plt.show()

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)


def main():
    print("\nLoading data...")
    X, y = load_iris_data('iris.txt')

    print("\nPerforming cross-validation...")
    scores = cross_validation(X, y)
    print("Cross-validation scores:", scores)
    print("Average accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("\nPerforming PCA and visualization...")
    pca_visualization(X, y)


if __name__ == "__main__":
    main()