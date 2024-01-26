from typing import List

#Input: nums = [1,2,3]
#Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = []
    n = len(nums)

    def backtrack(start, path):
        res.append(path)
        for i in range(start, n):
            # 分解成 剩余长度的子问题，
            backtrack(i + 1, path + [nums[i]])
    # b(0, []) = [] +  b(1, [1]) + b(2, [2]) + b(3, [3]),
    # b(1, [1]) 是包括1 的 所有子集，所以 只有 b(1, [1]) 是miss 不包括1 的自给
    # b(2, [2]) 是不包括1 包括2 的所有子集  b(3, [3]), 是不包括1 2 包括3 的所有子集。
    backtrack(0, [])
    return res

#Input: candidates = [2,3,6,7], target = 7
#Output: [[2,2,3],[7]]
# 是利用同样的组合搜索问题来执行一个function.
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res = []
    n = len(candidates)

    def backtrack(start, path, target):
        if target == 0:
            res.append(path)
            return
        for i in range(start, n):
            if target - candidates[i] < 0:
                continue
            backtrack(i, path + [candidates[i]], target - candidates[i])

    backtrack(0, [], target)
    return res

#Input: nums = [1,2,2]
#Output: [[],[1],[1,2],[1,2,2],[2],[2,2]], 不能出现两个 [1 2] [2] 的情况
def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    res = []
    n = len(nums)
    nums.sort()

    def backtrack(start, path):
        res.append(path)
        for i in range(start, n):
            # sort 之后，i > start 确保 第一个元素可以加入，[1,2,2]
            # 之后 如果当前元素和前一个元素相同，那么这个元素已经加过了，类比之前的 [1,2] [1,3] 的情况 都是在 b(1, [1])内的循环加入的
            if i > start and nums[i] == nums[i - 1]:
                continue
            backtrack(i + 1, path + [nums[i]])

    backtrack(0, [])
    return res

"""
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
"""
def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    res = []
    n = len(candidates)
    candidates.sort()

    def backtrack(start, path, target):
        if target == 0:
            res.append(path)
            return
        for i in range(start, n):
            if target - candidates[i] < 0:
                continue
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            backtrack(i + 1, path + [candidates[i]], target - candidates[i])

    backtrack(0, [], target)
    return res

#Input: nums = [1,2,3]
#Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
# 对比subset, 这里是全排列问题，所以1 需要skip已经访问过的元素。2 需要从头开始遍历。3是到达全部长度才记录
def permute(nums: List[int]) -> List[List[int]]:
    res = []
    n = len(nums)

    def backtrack(path):
        if len(path) == n:
            res.append(path)
            return
        for i in range(n):
            if nums[i] in path:
                continue
            backtrack(path + [nums[i]])

    backtrack([])
    return res

#Input: s = "aab"
#Output: [["a","a","b"],["aa","b"]]  every  substring of the partition is a palindrome
def partition(s):
    def is_palindrome(sub):
        return sub == sub[::-1]

    def backtrack(start, path):
        if start == len(s):
            ans.append(path)
            return
        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s[start:end]):
                backtrack(end, path + [s[start:end]])

    ans = []
    backtrack(0, [])
    return ans

# Input: digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
# 因为有order 所以是follow subset的pattern
def letterCombinations(digits: str) -> List[str]:
    if not digits:
        return []
    phone = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    res = []

    def backtrack(i, path):
        if i == len(digits):
            res.append(''.join(path))
            return
        for char in phone[digits[i]]:
            backtrack(i + 1, path + [char])

    backtrack(0, [])
    return res

def solveNQueens(n: int) -> List[List[str]]:
    def is_not_under_attack(row, col):
        return not (cols[col] + hill_diag[row - col] + dale_diag[row + col])

    def place_queen(row, col):
        queens[row] = col
        cols[col] = 1
        hill_diag[row - col] = 1
        dale_diag[row + col] = 1

    def remove_queen(row, col):
        queens[row] = 0
        cols[col] = 0
        hill_diag[row - col] = 0
        dale_diag[row + col] = 0

    def add_solution():
        solution = []
        for row in range(n):
            col = queens[row]
            solution.append('.' * col + 'Q' + '.' * (n - col - 1))
        output.append(solution)

    def backtrack(row):
        for col in range(n):
            if is_not_under_attack(row, col):
                place_queen(row, col)
                if row + 1 == n:
                    add_solution()
                else:
                    backtrack(row + 1)
                remove_queen(row, col)

    cols = [0] * n
    hill_diag = [0] * (2 * n - 1)
    dale_diag = [0] * (2 * n - 1)
    queens = [0] * n
    output = []
    backtrack(0)
    return output

def solveNQueensShort(n: int) -> List[List[str]]:
    def solve(row, cols, hills, dales):
        if row == n:
            board = ['.' * c + 'Q' + '.' * (n - c - 1) for c in cols]
            output.append(board)
        for col in range(n):
            # hill_diag[row - col] 检查所谓的“山”对角线（从左上到右下）是否已经有皇后。对于任意两个在同一“山”对角线上的格子，它们的 row - col 值是相同的。
            # dale_diag[row + col] 检查所谓的“谷”对角线（从右上到左下）是否已经有皇后。对于任意两个在同一“谷”对角线上的格子，它们的 row + col 值是相同的。
            if col not in cols and row - col not in hills and row + col not in dales:
                # 不用remove，因为是在call里面传递了新的list
                solve(row + 1, cols + [col], hills + [row - col], dales + [row + col])

    output = []
    solve(0, [], [], [])
    return output

def solveNQueensFast(n: int) -> List[List[str]]:
    def solve(row, cols, hills, dales):
        if row == n:
            board = ['.' * c + 'Q' + '.' * (n - c - 1) for c in cols]
            output.append(board)
            return
        for col in range(n):
            if not (cols[col] or hills[row - col] or dales[row + col]):
                cols[col] = hills[row - col] = dales[row + col] = 1
                solve(row + 1, cols, hills, dales)
                cols[col] = hills[row - col] = dales[row + col] = 0

    output = []
    cols = [0] * n  # 列状态
    hills = [0] * (2 * n)  # “山”对角线状态
    dales = [0] * (2 * n)  # “
    solve(0, cols, hills, dales)
    return output
