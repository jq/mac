from collections import defaultdict
from typing import List


def search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def search_2(nums: List[int], target: int) -> int:
    # user python built-in function
    try:
        return nums.index(target)
    except ValueError:
        return -1

def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid // n][mid % n]
        if mid_val == target:
            return True
        if mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

def minEatingSpeed(piles: List[int], h: int) -> int:
    def time_needed(speed):
        return sum((pile - 1) // speed + 1 for pile in piles)
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if time_needed(mid) > h:
            left = mid + 1
        else:
            right = mid
    return left

def findMin(nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        # 如果是正序 mid 必须小于 right，现在不是正序 说明最小值在mid 右边
        if nums[mid] > nums[right]:
            left = mid + 1
        # 是正序，最小值在mid 左边，mid也有可能是最小值
        elif nums[mid] < nums[right]:
            right = mid
        # 因为数组里面的值是unique 相等只可能 mid == right,
        # 又因为 mid = (left + right) // 2， 所以在没有重复值的情况不会走到该分支
        else:
            right -= 1
    # 没有重复值的时候 此时left == right
    return nums[left]

import bisect
from collections import defaultdict

def show_bisect():
    a = [1, 2, 4, 4, 5]
    x = 4
    #(a, x, lo=0, hi=len(a))
    index = bisect.bisect_left(a, x)
    print(f"bisect_left {index}")  # 插入到最左边保持有序的位置，是插入后的位置
    print(f"bisect_right {bisect.bisect_right(a, x)}")  # 插入到最右边保持有序的位置，是插入后的位置
    print(f"bisect {bisect.bisect(a, x)}")  # bisect_right
    bisect.insort(a, x)  #实际插入到右边 insort_right
    print(f"insort {a}")  # 输出: insort_right
    bisect.insort_left(a, x)
    print(f"insort_left {a}")  # 输出: None
def show_complex_bisect():
    people = [(25, 'John Doe'), (30, 'Jane Smith'), (35, 'Dave Brown')]
    print(f"bisect_left {bisect.bisect_left(people, (30, ))}")  # 输出: 1

show_complex_bisect()

class TimeMap:

    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    # 找到和timestamp最接近的时间戳的 对应的value
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        values = self.store[key]
        # only search timestamp
        # 从小到大排序，找到最右边的插入这个值的位置，假设有(timestamp, value) （timestapm,）就会找错位置
        # 因为没有chr(0x10FFFF) 会比任意字符小
        i = bisect.bisect_right(values, (timestamp,  chr(0x10FFFF))) #chr(255)也可以用
        return "" if i == 0 else values[i-1][1]

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    def get_kth_element(k):
        index1, index2 = 0, 0
        while True:
            if index1 == m:
                return nums2[index2 + k - 1]
            if index2 == n:
                return nums1[index1 + k - 1]
            if k == 1:
                return min(nums1[index1], nums2[index2])
            new_index1 = min(index1 + k // 2 - 1, m - 1)
            new_index2 = min(index2 + k // 2 - 1, n - 1)
            pivot1, pivot2 = nums1[new_index1], nums2[new_index2]
            if pivot1 <= pivot2:
                k -= new_index1 - index1 + 1
                index1 = new_index1 + 1
            else:
                k -= new_index2 - index2 + 1
                index2 = new_index2 + 1
    m, n = len(nums1), len(nums2)
    total_length = m + n
    if total_length % 2 == 1:
        return get_kth_element((total_length + 1) // 2)
    else:
        return (get_kth_element(total_length // 2) + get_kth_element(total_length // 2 + 1)) / 2