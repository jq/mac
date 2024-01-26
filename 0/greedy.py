# https://leetcode.com/problems/maximum-subarray/
import collections
from typing import List

# Given an integer array nums, find the subarray with the largest sum, and return its sum.
def maxSubArrayConst(nums: List[int]) -> int:
    current_sum = max_sum = nums[0]
    for i in range(1, len(nums)):
        # 除非current_sum 是负数，否则current_sum + nums[i] 一定比 nums[i] 大
        # 如果current_sum 是负数，那么就重新开始。
        # 加法的特性 所以局部最优，就是全局最优。n-1 的最优 和 n-1 + n 的最优比较 就得出n的最优。
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum

#https://leetcode.com/problems/jump-game/description/
# Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
# Each element in the array represents your maximum jump length at that position.
# Determine if you are able to reach the last index. Input: nums = [3,2,1,0,4] Output: False
def canJump(nums: List[int]) -> bool:
    # max_pos 是当前能到达的最远位置
    max_pos = 0
    for i, num in enumerate(nums):
        # 如果当前位置大于最远能到位置，那么就无法到达
        if i > max_pos:
            return False
        max_pos = max(max_pos, i + num)
    return True
# Return the minimum number of jumps to reach nums[n - 1].
#https://leetcode.com/problems/jump-game-ii/description/
def jump(nums: List[int]) -> int:
    n = len(nums)
    end = farthest = jumps = 0
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        # 如果i 到了之前跳到的最远位置，那么就需要再跳一次
        if i == end:
            jumps += 1
            # 更新最远位置
            end = farthest
    return jumps

#https://leetcode.com/problems/gas-station/description/
# There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
# You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1) station.
# You begin the journey with an empty tank at one of the gas stations.
# Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.
#If there exists a solution, it is guaranteed to be unique
"""总油量大于等于总消耗：只要整个路线上的总油量大于等于总的消耗量，就意味着肯定存在一个起点可以让你成功绕圈一周。这是因为，
不管你在哪个点的油量有多少剩余，只要总的输入（加的油）大于总的输出（油的消耗），就能通过调整起点来保证绕圈成功。"""
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    if sum(gas) < sum(cost):
        return -1
    start, tank = 0, 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            # 唯一性保证了 不用循环验证，如果没有唯一性 那么总油量》总消耗的条件，也是第一个找到正数到头的位置是start，因为之前的位置都是负数
            start, tank = i + 1, 0
    return start

#https://leetcode.com/problems/hand-of-straights/description/
# Alice has a hand of cards, given as an array of integers.
# Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.
# Return true if and only if she can.
"""Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
如果手中的牌的总数不能被W整除，那么显然无法分组，直接返回False。
使用一个计数器来统计每个数字出现的次数。
从最小的数字开始，尝试构建连续的数字组。对于每个数字组的起始数字m，检查从m到m+W-1的每个数字是否都存在。
如果某个数字不存在，说明无法形成连续组，返回False。
如果能为每个起始数字m成功构建连续组，则当计数器为空时，说明所有的牌都成功分组了，返回True。
"""
def isNStraightHand(hand: List[int], W: int) -> bool:
    if len(hand) % W != 0:
        return False
    count = collections.Counter(hand)
    while count:
        m = min(count)
        for k in range(m, m + W):
            v = count[k]
            if not v:
                return False
            if v == 1:
                del count[k]
            else:
                count[k] = v - 1
    return True

#https://leetcode.com/problems/merge-triplets-to-form-target-triplet/description/
# A triplet is an array of three integers. You are given a 2D integer array triplets, where triplets[i] = [ai, bi, ci]
# describes the ith triplet. You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.
# To obtain target, you may apply the following operation on triplets any number of times (possibly zero):
"""Choose two indices (0-indexed) i and j (i != j) and update triplets[j] to become [max(ai, aj), max(bi, bj), max(ci, cj)].
For example, if triplets[i] = [2, 5, 3] and triplets[j] = [1, 7, 5], triplets[j] will be updated to [max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5].
Return true if it is possible to obtain the target triplet [x, y, z] as an element of triplets, or false otherwise.

Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5]
Output: true
Explanation: Perform the following operations:
- Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. triplets = [[2,5,3],[1,8,4],[2,7,5]]
The target triplet [2,7,5] is now an element of triplets.
"""
def mergeTriplets(triplets: List[List[int]], target: List[int]) -> bool:
    res = [0, 0, 0]
    for a, b, c in triplets:
        # 遍历每个给定的三元组。对于每个三元组，只有当它的每个元素都不超过目标三元组对应位置的值时，才考虑将其用于更新res。
        if a <= target[0] and b <= target[1] and c <= target[2]:  # 步骤2
            res = [max(res[0], a), max(res[1], b), max(res[2], c)]  # 步骤3
    return res == target  # 步骤4

#https://leetcode.com/problems/partition-labels/description/
# A string S of lowercase English letters is given. We want to partition this string into as many parts as possible
# so that each letter appears in at most one part, and return a list of integers representing the size of these parts.
"""Input: s = "ababcbacadefegdehijhklij" 分割成最多的substring, 字母只出现在一个substring中，最少就是 1个 substring，
Output: [9,7,8]  output size of substring
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".  """
def partitionLabels(s: str) -> List[int]:
    #首先，遍历字符串，记录每个字符最后出现的位置。
    last = {c: i for i, c in enumerate(s)}
    start = end = 0
    res = []
    for i, c in enumerate(s):
        #当前片段中任何字符最后出现的最大位置
        end = max(end, last[c])
        # 遍历的位置i等于end，说明到达了可以分割的位置，因为之后的字符都不会出现在当前片段中。
        if i == end: # 因为 end = max(end, last[c]) 所以i 不可能 > end.
            res.append(end - start + 1)
            start = i + 1
    return res

#https://leetcode.com/problems/valid-parenthesis-string/description/
# Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
# The following rules define a valid string:
# Any left parenthesis '(' must have a corresponding right parenthesis ')'.
# Any right parenthesis ')' must have a corresponding left parenthesis '('.
# Left parenthesis '(' must go before the corresponding right parenthesis ')'.
# '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string. Input: s = "(*))"
# Output: true
# 使用return low <= 0并去掉low = max(low, 0) 有问题
# (**(
def checkValidString(s: str) -> bool:
    # low和high分别代表在考虑*为左括号和右括号时，可能的最小和最大未匹配左括号的数量
    low = high = 0
    for char in s:
        low += 1 if char == '(' else -1  # 视`*`为右括号 是最小匹配，时减少左括号计数
        high += 1 if char != ')' else -1  # 视`*`为左括号 是最大匹配 时增加左括号计数
        if high < 0:  # 如果high在任何时刻小于0，说明右括号太多 就是* 都是左括号，还不够
            return False
        # 提前计算的) 不能给以后得 ( 用，所以需要清零
        low = max(low, 0)  # 防止low变成负数 可以有* 变成空，每个点都需要保证正确 (**( 这种情况 没有每次清零就会出错

    return low == 0  # 确保所有左括号都被匹配
# 如果用 return low <= 0并去除low = max(low, 0) 出错，因为每个个位置都需要保证正确，
# 会忽略一种情况：在字符串的某个中间阶段，即使low暂时为负（表示右括号充足），之后仍可能出现过多的左括号无法被匹配。