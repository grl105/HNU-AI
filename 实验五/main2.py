import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


def pca_visualization(X, y):
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 创建和训练SVM分类器
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_pca, y)

    # 创建网格
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 可视化
    plt.figure(figsize=(10, 8))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolors='black')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('SVM Decision Boundary after PCA (sklearn)')

    plt.show()

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)


def main():
    print("\nLoading data...")
    X, y = load_iris_data('iris.txt')

    # 直接使用sklearn的交叉验证
    print("\nPerforming cross-validation...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    scores = cross_val_score(svm, X, y, cv=10)
    print("Cross-validation scores:", scores)
    print("Average accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("\nPerforming PCA and visualization...")
    pca_visualization(X, y)


if __name__ == "__main__":
    main()