# https://leetcode.com/problems/rotate-image/description/
#rotate the image in-place, rotate the image by 90 degrees (clockwise).
import collections
from typing import List

# 用 2*2 矩阵举例 转置矩阵 相当于对角线交换 列 满足要求，行不满足要求，然后再对每一行进行翻转。
def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    # 转置矩阵
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 反转每一行
    for i in range(n):
        matrix[i].reverse()

def transpose(matrix: List[List[int]]) -> List[List[int]]:
    # *matrix 把每行变成一个参数，zip 把相同位置的参数打包成元组， list 把元组打包成列表
    return list(zip(*matrix)) # [(1, 4, 7), (2, 5, 8), (3, 6, 9)] 不是原来的格式
    # [list(row) for row in zip(*matrix)] 是原来格式


print(transpose([[1,2,3],[4,5,6],[7,8,9]]))

#https://leetcode.com/problems/spiral-matrix/description/
def spiralOrder(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    res = []
    rows, cols = len(matrix), len(matrix[0])
    left, right, top, bottom = 0, cols - 1, 0, rows - 1
    while left <= right and top <= bottom:
        for col in range(left, right + 1):
            res.append(matrix[top][col])
        for row in range(top + 1, bottom + 1):
            res.append(matrix[row][right])
        #if left < right and top < bottom:的目的是防止在遍历完成一圈后，对单行或单列的矩阵进行重复遍历
        if left < right and top < bottom:
            for col in range(right - 1, left, -1):
                res.append(matrix[bottom][col])
            for row in range(bottom, top, -1):
                res.append(matrix[row][left])
        left += 1
        right -= 1
        top += 1
        bottom -= 1
    return res

#https://leetcode.com/problems/happy-number/description/
"""Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
"""
def isHappy(n: int) -> bool:
    def get_next(number):
        total_sum = 0
        while number > 0:
            number, digit = divmod(number, 10)
            total_sum += digit ** 2
        return total_sum

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)
    return n == 1

#https://leetcode.com/problems/plus-one/description/
def plusOne(digits: List[int]) -> List[int]:
    n = len(digits)

    for i in range(n-1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0

    # 如果所有位都是9，如[9, 9, 9]
    return [1] + [0] * n

#https://leetcode.com/problems/powx-n/description/
def myPow(x: float, n: int) -> float:
    if n < 0:
        x = 1 / x
        n = -n
    res = 1
    while n:
        if n & 1:
            res *= x
        x *= x
        n >>= 1
    return res


#https://leetcode.com/problems/multiply-strings/
def multiply(num1: str, num2: str) -> str:
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            # p1 是进位 p2 是当前位
            p1, p2 = i + j, i + j + 1
            sum = mul + res[p2] # 上一个的进位
            res[p1] += sum // 10
            res[p2] = sum % 10
    # 跳过前导0
    i = 0
    while i < len(res) and res[i] == 0:
        i += 1
    return ''.join(map(str, res[i:]))

# https://leetcode.com/problems/detect-squares/description/
# 新加的点和已有的点能构成几个正方形
class DetectSquares:

    def __init__(self):
        self.points = collections.defaultdict(int)

    def add(self, point: List[int]) -> None:
        self.points[tuple(point)] += 1

    def count(self, point: List[int]) -> int:
        square_count = 0
        x1, y1 = point

        for (x2, y2), n in self.points.items():
            # 检查它与输入点point是否能构成正方形的两个对角点  x1 - x2和y1 - y2的绝对值是否相等且不为0
            x_dist, y_dist = abs(x1 - x2), abs(y1 - y2)
            if x_dist == y_dist and x_dist > 0:
                corner1 = (x1, y2)
                corner2 = (x2, y1)
                if corner1 in self.points and corner2 in self.points:
                    square_count += n * self.points[corner1] * self.points[corner2]

        return square_count