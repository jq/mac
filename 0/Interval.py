#https://leetcode.com/problems/insert-interval/description/
import heapq
from typing import List

# Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
# Output: [[1,5],[6,9]]
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    merged = []
    i = 0
    #添加所有结束时间小于新区间开始时间的区间
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        merged.append(intervals[i])
        i += 1
    # 合并所有与新区间重叠的区间，开始时间小于等于新区间结束时间，结束时间大于等于新区间开始时间
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
        i += 1
    merged.append(newInterval)
    # 添加所有开始时间大于新区间结束时间的区间
    while i < len(intervals):
        merged.append(intervals[i])
        i += 1

    return merged


#https://leetcode.com/problems/merge-intervals/description/
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # 如果列表为空，或者当前区间与上一区间不重合，直接添加
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 与上一区间进行合并，尾部取最大值
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

#https://leetcode.com/problems/non-overlapping-intervals/
"""Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
策略：优先保留结束时间早的区间
通过每次选择结束时间最早的区间，我们能够为后续的区间留下尽可能多的空间，从而减少必须移除的区间数量。
假设当区间数量为k时，贪心策略能保证最少的移除数量，那么在区间数量为k+1时，通过选择结束时间最早的区间，
我们可以确保对于任何可能的第k+2个区间，都留有最大的空间，从而最小化移除数量。
"""
def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    right = intervals[0][1] #right表示当前不重叠区间的结束时间，初始化为第一个区间的结束时间
    removed = 0
    for i in range(1, len(intervals)):
        # 局部最优 = 全局最优的例子。
        if intervals[i][0] < right:
            removed += 1
        else:
            right = intervals[i][1]
    return removed

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

# https://www.lintcode.com/problem/920/
# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
# determine if a person could attend all meetings. 就是判断是否有重复
def can_attend_meetings(self, intervals: List[Interval]) -> bool:
    intervals.sort(key=lambda x: x.start)
    for i in range(1, len(intervals)):
        if intervals[i].start < intervals[i-1].end:
            return False
    return True


#https://www.lintcode.com/problem/919/
"""Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
find the minimum number of conference rooms required.)
Input: intervals = [(0,30),(5,10),(15,20)] output: 2
是要找到最大的重叠 =》检查在任何给定时间点，有多少会议正在进行
使用一个计数器来表示当前正在进行的会议数量。每当我们遇到一个开始时间，就增加计数器；每当遇到一个结束时间，就减少计数器。
这样，计数器的最大值就是同时进行的最大会议数量，也就是我们需要的最小会议室数量。
"""
def min_meeting_rooms(intervals: List[Interval]) -> int:
    starts = sorted([i.start for i in intervals])
    ends = sorted([i.end for i in intervals])

    s, e = 0, 0
    max_rooms = 0
    current_rooms = 0

    # 不需要判断 e 因为 如果s 一致向前 e 不动，那么已经增加了会议室。
    while s < len(starts):
        if starts[s] < ends[e]:
            current_rooms += 1
            s += 1
        else:
            current_rooms -= 1
            e += 1
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms

#https://leetcode.com/problems/minimum-interval-to-include-each-query/description/
"""The answer to the jth query is the size of the smallest interval i such that lefti <= queries[j] <= righti. 
If no such interval exists, the answer is -1.
Input: intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]
Output: [3,3,1,4]
Explanation: The queries are processed as follows:
- Query = 2: The interval [2,4] is the smallest interval containing 2. The answer is 4 - 2 + 1 = 3.
- Query = 3: The interval [2,4] is the smallest interval containing 3. The answer is 4 - 2 + 1 = 3.
- Query = 4: The interval [4,4] is the smallest interval containing 4. The answer is 4 - 4 + 1 = 1.
- Query = 5: The interval [3,6] is the smallest interval containing 5. The answer is 6 - 3 + 1 = 4.
遍历每个查询，同时将所有起始位置小于等于当前查询值的区间加入到一个优先队列（最小堆）中。在这个堆中，区间按照它们的大小（即区间的长度）进行排序。
"""
def minInterval(intervals: List[List[int]], queries: List[int]) -> List[int]:
    intervals.sort()
    queries = sorted([(q, i) for i, q in enumerate(queries)])
    res = [-1] * len(queries)
    heap = []
    i = 0
    for q, idx in queries:
        while i < len(intervals) and intervals[i][0] <= q:
            size = intervals[i][1] - intervals[i][0] + 1
            heapq.heappush(heap, (size, intervals[i][1]))
            i += 1
        while heap and heap[0][1] < q:
            heapq.heappop(heap)
        if heap:
            res[idx] = heap[0][0]
    return res