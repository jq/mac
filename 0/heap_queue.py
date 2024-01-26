import collections
from random import random
from typing import List

"""
Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]
"""
import heapq

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)  # 将nums转换为最小堆
        while len(self.heap) > k:  # 保持堆的大小为k
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:  # 只有当新元素大于堆顶元素时，才有可能成为第k大元素
            # 将新元素加入堆，并弹出堆顶元素
            heapq.heappushpop(self.heap, val)
        return self.heap[0]  # 堆顶元素是第k大的元素

#heapq模块并没有直接提供最大堆的实现。但是，你可以通过存储每个数的相反数来间接实现一个最大堆。
def lastStoneWeight(stones: List[int]) -> int:
    heap = [-x for x in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        x, y = heapq.heappop(heap), heapq.heappop(heap)
        if x != y:
            heapq.heappush(heap, x - y)
    return -heap[0] if heap else 0

def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    return heapq.nsmallest(k, points, key=lambda x: x[0] ** 2 + x[1] ** 2)

#https://leetcode.com/problems/kth-largest-element-in-an-array/description/
def findKthLargest(nums: List[int], k: int) -> int:
    return heapq.nlargest(k, nums)[-1]

def findKthLargest2(nums: List[int], k: int) -> int:
    left, right = 0, len(nums) - 1
    while True:
        pivot_index = random.randint(left, right)
        new_pivot_index = partition(nums, left, right, pivot_index)
        if new_pivot_index == len(nums) - k:
            return nums[new_pivot_index]
        elif new_pivot_index > len(nums) - k:
            right = new_pivot_index - 1
        else:
            left = new_pivot_index + 1
# 划分函数重新排列数组中的元素，使得小于轴心点的元素来到轴心点的前面，而大于轴心点的元素则来到其后面。
def partition(nums, left, right, pivot_index):
    pivot = nums[pivot_index]
    nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
    stored_index = left
    for i in range(left, right):
        if nums[i] < pivot:
            nums[i], nums[stored_index] = nums[stored_index], nums[i]
            stored_index += 1
    nums[right], nums[stored_index] = nums[stored_index], nums[right]
    return stored_index

#https://leetcode.com/problems/task-scheduler/  ["A","A","A","B","B","B"], n = 2
# After completing task A, you must wait two cycles before doing A again.
# 每一组最高频率的任务之间需要有n个单位时间的间隔。然后，我们在这些间隔中填充其他任务。
#如果任务种类不够多，那么我们需要等待，直到满足再次执行最高频率任务的冷却时间。
def leastInterval(tasks: List[str], n: int) -> int:
    task_counts = collections.Counter(tasks).values()
    max_count = max(task_counts)
    # 有几个task 有max count
    max_count_tasks = sum(count == max_count for count in task_counts)
    # 一种是冷却期需要被用上，也就是中间的填充不够多：
    # 如果一个任务执行了 max_count 次，那么它之间只需要 max_count - 1 个冷却期
    # 对于每个冷却期，我们有 n 个单位时间的间隔，加上执行一次任务本身需要的1个单位时间，所以每个周期（一次任务执行加上冷却时间）是 n + 1 个单位时间。
    # 因为  max_count - 1 个冷却期 少算了最后一个执行， max_count_tasks 是在最后一轮中执行所有频率最高的任务（因为它们可以同时或连续执行，不需要再等待冷却时间）
    # 如果冷却前无需使用，那么就是 len(tasks)
    return max(len(tasks), (max_count - 1) * (n + 1) + max_count_tasks)

# https://leetcode.com/problems/find-median-from-data-stream/description/
class MedianFinder:
    def __init__(self):
        self.small = []  # store the smaller half of the numbers
        self.large = []  # store the larger half of the numbers

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        if len(self.large) < len(self.small):
            heapq.heappush(self.large, -heapq.heappop(self.small))

    def findMedian(self) -> float:
        if len(self.large) > len(self.small):
            return float(self.large[0])
        return (self.large[0] - self.small[0]) / 2.0
