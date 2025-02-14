import mindspore
import mindspore.nn as nn
import numpy as np


class EightQueensSolver:
    def __init__(self, board_size=8, initial_board=None, max_solutions=5):
        self.board_size = board_size
        self.solutions = []
        self.max_solutions = max_solutions

        if initial_board is None:
            self.board = [-1] * board_size
        else:
            if len(initial_board) != board_size:
                raise ValueError(f"初始棋盘长度必须为 {board_size}")
            self.board = initial_board.copy()

    def is_safe(self, board, row, col):
        # 检查当前行之前的所有行
        for i in range(row):
            # 只检查已经放置了皇后的位置
            if board[i] != -1:
                # 检查是否在同一列
                if board[i] == col:
                    return False

                # 检查对角线
                # 两点在对角线上的条件：行之差的绝对值 = 列之差的绝对值
                row_diff = row - i
                col_diff = abs(col - board[i])
                if row_diff == col_diff:
                    return False
        return True

    def solve_n_queens(self):
        """解决八皇后问题"""
        # 找到第一个未放置皇后的行
        start_row = 0
        while start_row < self.board_size and self.board[start_row] != -1:
            start_row += 1

        self.backtrack(self.board, start_row)
        return self.solutions

    def backtrack(self, board, row):
        """使用回溯法求解"""
        # 如果已经超过最大解数量，停止搜索
        if len(self.solutions) >= self.max_solutions:
            return

        # 如果已经处理完所有行，说明找到一个解
        if row == self.board_size:
            self.solutions.append(board.copy())
            return

        # 在当前行尝试每一列
        for col in range(self.board_size):
            if self.is_safe(board, row, col):
                board[row] = col  # 放置皇后
                self.backtrack(board, row + 1)  # 递归处理下一行
                board[row] = -1  # 回溯，撤销当前选择

    def print_solution(self, solution, solution_num=None):
        if solution_num is not None:
            print(f"\n解决方案 {solution_num}:")

        for row in range(self.board_size):
            line = []
            for col in range(self.board_size):
                line.append('👑' if solution[row] == col else '⬜')
            print(' '.join(line))

    def print_all_solutions(self):
        """打印所有找到的解决方案（最多5个）"""
        if not self.solutions:
            print("未找到解决方案！")
            return

        print(f"\n找到 {len(self.solutions)} 个解决方案:")
        for idx, solution in enumerate(self.solutions, 1):
            self.print_solution(solution, idx)
            print()


def main():
    # 示例1：默认空棋盘
    print("示例1：默认空棋盘")
    solver1 = EightQueensSolver()
    solver1.solve_n_queens()
    print(f"找到的解决方案数量: {len(solver1.solutions)}")
    solver1.print_all_solutions()

    # 示例2：自定义初始棋盘
    print("\n示例2：自定义初始棋盘")
    initial_board = [-1] * 8
    initial_board[0] = 0  # 第一行第一列
    initial_board[1] = 4
    #initial_board[2] = 1
    #initial_board[3] = 5
    #initial_board[4] = 2
    #initial_board[6] = 3
    #initial_board[7] = 7
    solver2 = EightQueensSolver(initial_board=initial_board)
    solver2.solve_n_queens()
    print(f"找到的解决方案数量: {len(solver2.solutions)}")
    solver2.print_all_solutions()


if __name__ == "__main__":
    main()