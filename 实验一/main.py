import mindspore
import mindspore.nn as nn
import heapq


class SearchNode:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # 从起点到当前节点的代价
        self.h = h  # 启发式估计代价
        self.f = g + h  # 总代价

    def __lt__(self, other):
        return self.f < other.f


class RomaniaMap:
    def __init__(self):
        # 定义罗马尼亚地图的图结构
        self.graph = {
            'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
            'Zerind': {'Arad': 75, 'Oradea': 71},
            'Oradea': {'Zerind': 71, 'Sibiu': 151},
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
            'Timisoara': {'Arad': 118, 'Lugoj': 111},
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
            'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
            'Drobeta': {'Mehadia': 75, 'Craiova': 120},
            'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
            'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
            'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
            'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
            'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
            'Giurgiu': {'Bucharest': 90},
            'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
            'Hirsova': {'Urziceni': 98, 'Eforie': 86},
            'Eforie': {'Hirsova': 86},
            'Vaslui': {'Urziceni': 142, 'Iasi': 92},
            'Iasi': {'Vaslui': 92, 'Neamt': 87},
            'Neamt': {'Iasi': 87}
        }

        # 启发式估计（直线距离）
        self.heuristics = {
            'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
            'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
            'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
            'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
            'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199,
            'Zerind': 374
        }

    def heuristic(self, state):
        return self.heuristics.get(state, float('inf'))

    def get_neighbors(self, state):
        return self.graph.get(state, {})


def astar_search(map_instance, start, goal):
    start_node = SearchNode(start, g=0, h=map_instance.heuristic(start))
    frontier = []
    heapq.heappush(frontier, start_node)

    # 使用字典来跟踪最佳路径
    came_from = {start: None}

    # 记录到每个节点的最小代价
    cost_so_far = {start: 0}

    while frontier:
        current_node = heapq.heappop(frontier)

        if current_node.state == goal:
            # 重建路径
            path = []
            current = current_node.state
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]

        for neighbor, cost in map_instance.get_neighbors(current_node.state).items():
            new_cost = cost_so_far[current_node.state] + cost

            # 如果邻居节点之前未访问，或找到了更优路径
            if (neighbor not in cost_so_far or
                    new_cost < cost_so_far[neighbor]):
                cost_so_far[neighbor] = new_cost
                h_cost = map_instance.heuristic(neighbor)
                priority = new_cost + h_cost

                neighbor_node = SearchNode(
                    neighbor,
                    parent=current_node,
                    g=new_cost,
                    h=h_cost
                )

                heapq.heappush(frontier, neighbor_node)
                came_from[neighbor] = current_node.state

    return None


# 主程序
def main():
    # 创建罗马尼亚地图
    romania_map = RomaniaMap()

    # 设置起点和终点
    start = 'Arad'
    goal = 'Iasi'

    # 执行A*搜索
    path = astar_search(romania_map, start, goal)

    # 输出结果
    print(f"从 {start} 到 {goal} 的最短路径:")

    # 调试：打印完整路径信息
    print("完整路径:", path)

    # 输出路径
    print(" -> ".join(path))

    # 时间复杂度分析
    print("\n时间复杂度分析:")
    print("最坏情况时间复杂度: O(b^d)")
    print("其中b是分支因子，d是解的深度")
    print("空间复杂度: O(b^d)")


if __name__ == "__main__":
    main()