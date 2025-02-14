import numpy as np
import pandas as pd
from mindspore import context
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class OptimizedCARTDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def calc_gini(self, y):
        """计算基尼指数"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def find_best_split(self, X, y):
        """找到最佳分裂特征和分裂点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        parent_gini = self.calc_gini(y)

        # 记录所有可能的分裂点
        potential_splits = []

        for feature in X.columns:
            # 对于每个特征，找最佳分裂点
            unique_values = np.unique(X[feature])

            for threshold in unique_values:
                # 连续特征分裂
                left_mask = X[feature] <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                # 加权基尼指数
                gini_left = self.calc_gini(y_left)
                gini_right = self.calc_gini(y_right)
                split_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                # 基尼增益
                gini_gain = parent_gini - split_gini

                potential_splits.append({
                    'feature': feature,
                    'threshold': threshold,
                    'gain': gini_gain,
                    'left_mask': left_mask,
                    'right_mask': right_mask
                })

        # 如果没有有效分裂
        if not potential_splits:
            return None, None, 0

        # 选择增益最大的分裂
        best_split = max(potential_splits, key=lambda x: x['gain'])

        return (best_split['feature'],
                best_split['threshold'],
                best_split['gain'],
                best_split['left_mask'],
                best_split['right_mask'])

    def _build_tree(self, X, y, depth=0):
        # 停止条件
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            # 返回出现最多的类别
            most_common = np.bincount(y).argmax()
            return {'prediction': most_common}

        # 寻找最佳分裂
        split_result = self.find_best_split(X, y)

        # 无法分裂
        if split_result[0] is None:
            most_common = np.bincount(y).argmax()
            return {'prediction': most_common}

        # 解包分裂结果
        best_feature, best_threshold, gain, left_mask, right_mask = split_result

        # 合并相似的子节点
        y_left = y[left_mask]
        y_right = y[right_mask]

        # 检查子节点是否可以合并
        left_majority = np.bincount(y_left).argmax()
        right_majority = np.bincount(y_right).argmax()

        # 如果子节点类别相同，直接返回多数类
        if left_majority == right_majority:
            return {'prediction': left_majority}

        # 构建树
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': gain
        }

        # 递归构建子树
        tree['left'] = self._build_tree(
            X[left_mask], y[left_mask], depth + 1
        )
        tree['right'] = self._build_tree(
            X[right_mask], y[right_mask], depth + 1
        )

        return tree

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._predict_single(row, self.tree) for _, row in X.iterrows()])

    def _predict_single(self, x, tree):
        if 'prediction' in tree:
            return tree['prediction']

        feature_value = x[tree['feature']]
        if feature_value <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


def plot_cart_tree(tree, ax):
    """绘制完整的CART决策树"""
    decision_node_color = '#B3E5FC'  # 浅蓝色决策节点
    leaf_node_high_color = '#C8E6C9'  # 浅绿色高销量节点
    leaf_node_low_color = '#FFCDD2'  # 浅红色低销量节点

    def draw_node(x, y, content, node_type='decision'):
        """绘制节点"""
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
        """绘制连接线"""
        ax.plot([start[0], end[0]], [start[1], end[1]], 'black',
                linestyle='--', zorder=1)
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.1, text, ha='center', va='bottom',
                    color='black', fontsize=10)

    def draw_tree(node, x, y, dx):
        """递归绘制树"""
        if 'prediction' in node:
            # 叶节点
            value = "HIGH" if node['prediction'] == 1 else "LOW"
            node_type = 'high' if node['prediction'] == 1 else 'low'
            draw_node(x, y, f"Sales:\n{value}", node_type)
            return

        # 决策节点
        feature_name = node['feature']
        threshold = node['threshold']
        gini = node['gain']
        draw_node(x, y, f"{feature_name}<={threshold}\nGini={gini:.3f}")

        # 递归绘制左子树
        if 'left' in node:
            next_x = x - dx
            next_y = y - 1
            draw_connection((x, y), (next_x, next_y), 'Yes')
            draw_tree(node['left'], next_x, next_y, dx / 2)

        # 递归绘制右子树
        if 'right' in node:
            next_x = x + dx
            next_y = y - 1
            draw_connection((x, y), (next_x, next_y), 'No')
            draw_tree(node['right'], next_x, next_y, dx / 2)

    # 设置图形属性
    ax.set_title('CART Decision Tree', pad=20, size=14)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 从根节点开始递归绘制整棵树
    draw_tree(tree, 0, 4, 3)

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

    # 训练CART决策树
    cart_tree = OptimizedCARTDecisionTree()
    cart_tree.fit(X, y)

    # 预测和计算准确率
    y_pred = cart_tree.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # 创建决策树可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_cart_tree(cart_tree.tree, ax)

    plt.savefig('optimized_cart_decision_tree.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 输出准确率
    print(f"\nCART决策树准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()