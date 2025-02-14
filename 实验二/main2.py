import mindspore
import mindspore.nn as nn
import numpy as np


class EightQueensSolver:
    def __init__(self, board_size=8, initial_board=None, max_solutions=1):
        self.board_size = board_size
        self.solutions = []
        self.max_solutions = max_solutions
        self.step_count = 0

        if initial_board is None:
            self.board = [-1] * board_size
        else:
            if len(initial_board) != board_size:
                raise ValueError(f"初始棋盘长度必须为 {board_size}")
            self.board = initial_board.copy()

    def is_safe(self, board, row, col):
        for i in range(row):
            if board[i] != -1:
                if board[i] == col:
                    return False
                row_diff = row - i
                col_diff = abs(col - board[i])
                if row_diff == col_diff:
                    return False
        return True

    def solve_n_queens(self):
        self.step_count = 0
        start_row = 0
        while start_row < self.board_size and self.board[start_row] != -1:
            start_row += 1

        print("\n开始求解过程：")
        print("初始状态：")
        self.print_solution(self.board)
        print("\n")

        self.backtrack(self.board, start_row)
        return self.solutions

    def backtrack(self, board, row):
        if len(self.solutions) >= self.max_solutions:
            return

        if row == self.board_size:
            self.solutions.append(board.copy())
            return

        for col in range(self.board_size):
            if self.is_safe(board, row, col):
                board[row] = col
                self.step_count += 1

                print(f"步骤 {self.step_count}:")
                print(f"在第 {row} 行，第 {col} 列放置皇后")
                self.print_solution(board)
                print("\n")

                self.backtrack(board, row + 1)

                if self.solutions:
                    return

                board[row] = -1
                print(f"回溯：移除第 {row} 行，第 {col} 列的皇后")
                self.print_solution(board)
                print("\n")

    def print_solution(self, solution, solution_num=None):
        if solution_num is not None:
            print(f"\n解决方案 {solution_num}:")

        print("   " + " ".join(str(i) for i in range(self.board_size)))

        for row in range(self.board_size):
            print(f"{row}: ", end="")
            line = []
            for col in range(self.board_size):
                line.append('👑' if solution[row] == col else '⬜')
            print(' '.join(line))


def main():
    '''# 示例1：默认空棋盘
    print("示例1：默认空棋盘")
    solver1 = EightQueensSolver()
    solver1.solve_n_queens()
    print(f"找到的解决方案数量: {len(solver1.solutions)}")'''

    # 示例2：自定义初始棋盘
    print("\n示例2：自定义初始棋盘")
    initial_board = [-1] * 8
    initial_board[0] = 0  # 第一行第一列
    initial_board[1] = 4
    #initial_board[2] = 1
    #initial_board[3] = 5
    # initial_board[4] = 2
    # initial_board[6] = 3
    # initial_board[7] = 7

    solver2 = EightQueensSolver(initial_board=initial_board)
    solver2.solve_n_queens()
    print(f"找到的解决方案数量: {len(solver2.solutions)}")


if __name__ == "__main__":
    main()