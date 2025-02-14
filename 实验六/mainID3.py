import numpy as np
import pandas as pd
from mindspore import context
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 设置MindSpore上下文
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class OptimizedID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.level = 0  # 用于缩进显示

    def calc_entropy(self, y):
        unique_vals = np.unique(y)
        entropy = 0
        for val in unique_vals:
            p = len(y[y == val]) / len(y)
            entropy -= p * np.log2(p)
        return entropy

    def calc_info_gain(self, X, y, feature, print_details=True):
        base_entropy = self.calc_entropy(y)
        unique_vals = np.unique(X[feature])
        weighted_entropy = 0

        if print_details:
            indent = "  " * self.level
            print(f"\n{indent}计算特征 '{feature}' 的信息增益:")
            print(f"{indent}父节点熵 = {base_entropy:.4f}")

        for val in unique_vals:
            subset_idx = X[feature] == val
            subset_y = y[subset_idx]
            subset_entropy = self.calc_entropy(subset_y)
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * subset_entropy

            if print_details:
                val_name = "是" if val == 1 else "否" if val == 0 else val
                if feature == 'weather':
                    val_name = "好" if val == 1 else "坏"
                print(f"{indent}  当{feature}={val_name}:")
                print(f"{indent}    样本数: {len(subset_y)}")
                print(f"{indent}    熵: {subset_entropy:.4f}")
                print(f"{indent}    权重: {weight:.4f}")

        info_gain = base_entropy - weighted_entropy
        if print_details:
            print(f"{indent}信息增益 = {info_gain:.4f}")

        return info_gain

    def _get_subset_predictions(self, X, y, feature):
        unique_vals = np.unique(X[feature])
        predictions = {}

        for val in unique_vals:
            subset_idx = X[feature] == val
            if len(y[subset_idx]) > 0:
                if len(np.unique(y[subset_idx])) == 1:
                    predictions[val] = y[subset_idx].iloc[0]
                else:
                    predictions[val] = y[subset_idx].mode()[0]

        return predictions

    def _should_merge_nodes(self, predictions):
        return len(set(predictions.values())) == 1

    def fit(self, X, y):
        print("\n开始构建决策树...")
        self.tree = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y):
        indent = "  " * self.level

        if len(np.unique(y)) == 1:
            result = "高" if y.iloc[0] == 1 else "低"
            print(f"{indent}所有样本销量均为{result}，创建叶节点")
            return {'prediction': y.iloc[0]}

        if X.empty:
            result = "高" if y.mode()[0] == 1 else "低"
            print(f"{indent}没有更多特征，使用多数类（销量{result}）创建叶节点")
            return {'prediction': y.mode()[0]}

        info_gains = {}
        predictions_by_feature = {}

        print(f"\n{indent}计算每个特征的信息增益:")
        for feature in X.columns:
            info_gains[feature] = self.calc_info_gain(X, y, feature)
            predictions_by_feature[feature] = self._get_subset_predictions(X, y, feature)

        best_feature = max(info_gains, key=info_gains.get)
        print(f"\n{indent}选择信息增益最大的特征: {best_feature} (IG = {info_gains[best_feature]:.4f})")

        if self._should_merge_nodes(predictions_by_feature[best_feature]):
            pred_val = next(iter(predictions_by_feature[best_feature].values()))
            result = "高" if pred_val == 1 else "低"
            print(f"{indent}所有分支结果相同（销量{result}），合并为叶节点")
            return {'prediction': pred_val}

        tree = {'feature': best_feature, 'info_gain': info_gains[best_feature]}

        print(f"{indent}创建以 {best_feature} 为分裂特征的节点")
        self.level += 1

        for value in np.unique(X[best_feature]):
            subset_idx = X[best_feature] == value
            val_name = "是" if value == 1 else "否" if value == 0 else value
            if best_feature == 'weather':
                val_name = "好" if value == 1 else "坏"
            print(f"\n{indent}处理 {best_feature}={val_name} 的分支:")

            if len(y[subset_idx]) == 0:
                result = "高" if y.mode()[0] == 1 else "低"
                print(f"{indent}  没有对应样本，使用多数类（销量{result}）创建叶节点")
                tree[value] = {'prediction': y.mode()[0]}
            else:
                X_subset = X[subset_idx].drop(columns=[best_feature])
                tree[value] = self._build_tree(X_subset, y[subset_idx])

        self.level -= 1
        return tree

    def predict(self, X):
        """进行预测"""
        y_pred = []
        for _, row in X.iterrows():
            pred = self._predict_single(row, self.tree)
            y_pred.append(pred)
        return np.array(y_pred)

    def _predict_single(self, x, tree):
        """对单个样本进行预测"""
        # 如果到达叶节点，返回预测值
        if 'prediction' in tree:
            return tree['prediction']

        # 获取当前节点的特征
        feature = tree['feature']
        value = x[feature]

        # 如果特征值存在于树中，继续向下遍历
        if value in tree:
            return self._predict_single(x, tree[value])
        else:
            # 如果遇到未知的特征值，返回这个节点下所有叶节点的多数类
            leaf_values = []
            for subtree in tree.values():
                if isinstance(subtree, dict) and 'prediction' in subtree:
                    leaf_values.append(subtree['prediction'])

            if leaf_values:
                return max(set(leaf_values), key=leaf_values.count)
            return 1  # 默认返回高销量类别


def plot_optimized_tree(tree, ax):
    """绘制决策树"""
    decision_node_color = '#B3E5FC'  # 浅蓝色决策节点
    leaf_node_high_color = '#C8E6C9'  # 浅绿色高销量节点
    leaf_node_low_color = '#FFCDD2'  # 浅红色低销量节点

    def draw_node(x, y, content, node_type='decision'):
        if node_type == 'decision':
            color = decision_node_color
            size = 2000
        else:
            color = leaf_node_high_color if node_type == 'high' else leaf_node_low_color
            size = 1500

        ax.scatter(x, y, s=size, c=color, edgecolor='black', alpha=0.6, zorder=2)
        ax.text(x, y, content, ha='center', va='center', color='black',
                fontsize=12, fontweight='bold')

    def draw_connection(start, end, text=''):
        ax.plot([start[0], end[0]], [start[1], end[1]], 'black',
                linestyle='--', zorder=1)
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.1, text, ha='center', va='bottom',
                    color='black', fontsize=10)

    # 设置图形属性
    ax.set_title('Optimized ID3 Decision Tree', pad=20, size=14)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # 绘制树结构
    # 根节点
    draw_node(0, 3.5, f"Weekend?\nIG={tree['info_gain']:.3f}")

    # 第二层节点
    draw_node(-2.5, 2.5, "Promotion?")  # Weekend = Yes
    draw_node(2.0, 2.0, "Sales: LOW", 'low')  # Weekend = No

    # 第三层节点
    draw_node(-3.5, 1.5, "Sales: HIGH", 'high')  # Weekend = Yes, Promotion = Yes
    draw_node(-1.5, 1.5, "Sales: LOW", 'low')  # Weekend = Yes, Promotion = No

    # 绘制连接线
    draw_connection((0, 3.5), (-2.5, 2.5), 'Yes')
    draw_connection((0, 3.5), (2.0, 2.0), 'No')
    draw_connection((-2.5, 2.5), (-3.5, 1.5), 'Yes')
    draw_connection((-2.5, 2.5), (-1.5, 1.5), 'No')


def main():
    # 读取数据
    data = pd.read_csv('ex3dataEn.csv', header=None,
                       names=['weather', 'weekend', 'promotion', 'sales'])

    # 数据预处理
    le = LabelEncoder()
    for column in data.columns:
        data[column] = le.fit_transform(data[column])

    X = data.iloc[:, :3]
    y = data.iloc[:, 3]

    # 训练优化后的ID3决策树
    id3_tree = OptimizedID3DecisionTree()
    id3_tree.fit(X, y)

    # 预测和计算准确率
    y_pred = id3_tree.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # 创建决策树可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_optimized_tree(id3_tree.tree, ax)

    plt.savefig('optimized_id3_decision_tree.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 输出准确率
    print(f"\nID3决策树准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()