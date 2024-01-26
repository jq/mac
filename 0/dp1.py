#https://leetcode.com/problems/climbing-stairs/description/
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# dp[i] = dp[i-1] + dp[i-2]
from typing import List


def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    prev, current = 1, 2  # 初始化前两个台阶的方法数
    for i in range(3, n + 1):
        prev, current = current, prev + current  # 更新方法数
    return current

#https://leetcode.com/problems/min-cost-climbing-stairs/description/
# You can either start from the step with index 0, or the step with index 1.
"""顶部实际上是在给定的cost数组之外的。换句话说，你可以选择从索引为n-1的台阶向上爬一步到达顶部，或者从索引为n-2的台阶向上爬两步到达顶部。
因此，最终的最小花费应该是从这两个台阶中选择一个花费较小的方案。"""
def minCostClimbingStairs(cost: List[int]) -> int:
    n = len(cost)
    dp = [0] * n
    dp[0], dp[1] = cost[0], cost[1]
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
    return min(dp[-1], dp[-2])

def minCostClimbingStairsConst(cost):
    n = len(cost)
    # 初始化为到达第0步和第1步的成本
    first, second = cost[0], cost[1]
    for i in range(2, n):
        # 计算到达当前步的最小成本
        current = cost[i] + min(first, second)
        # 更新前两步的成本
        first, second = second, current
    # 返回到达最后一步或倒数第二步的最小成本
    return min(first, second)

#https://leetcode.com/problems/house-robber/description/
# no adjacent allowed
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
def rob(nums: List[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    if n == 1:
        return nums[0]
    dp = [0] * n
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[-1]

def robConst(nums):
    rob1, rob2 = 0, 0

    # Iterate through the house values
    for n in nums:
        # Calculate the new maximum amount considering two scenarios:
        # 1. Robbing the current house and the amount obtained from robbing two houses ago.
        # 2. Not robbing the current house and maintaining the previous maximum.
        newRob = max(rob1 + n, rob2)
        rob1 = rob2
        rob2 = newRob
    return rob2

#https://leetcode.com/problems/house-robber-ii/
# rob1 arranged in a circle.
# Return the maximum of three scenarios:
# 1. Robbing the first house and skipping the last house.
# 2. Robbing the last house and skipping the first house.
# 3 nums[0] 是 长度=1 的case
def rob2(nums: List[int]) -> int:
    return  max(nums[0], robConst(nums[1:]), robConst(nums[:-1]))

#https://leetcode.com/problems/longest-palindromic-substring/description/
# Given a string s, return the longest palindromic substring in s. Input: s = "babad"
# Output: "bab"
# https://leetcode.com/problems/longest-palindromic-substring/solutions/4212564/beats-96-49-5-different-approaches-brute-force-eac-dp-ma-recursion/
def longestPalindrome(s: str) -> str:
    def expandAroundCenter(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    if len(s) < 1:
        return ""
    res = ""
    for i in range(len(s)):
        s1 = expandAroundCenter(i, i)
        s2 = expandAroundCenter(i, i + 1)
        # max 的 key 用法
        res = max(res, s1, s2, key=len)
    return res

#https://leetcode.com/problems/palindromic-substrings/description/
# Given a string, your task is to count how many palindromic substrings in this string.
#可能的回文中心有2n-1个（包括每个字符和每两个字符之间的位置）。对于每个可能的回文中心，我们尝试向左右扩展，以检查以该中心为中心的最长回文子串。
def countSubstrings(s: str) -> int:
    n = len(s)
    ans = 0
    for center in range(2 * n - 1):
        left = center // 2
        right = left + center % 2
        while left >= 0 and right < n and s[left] == s[right]:
            ans += 1
            left -= 1
            right += 1
    return ans

#https://leetcode.com/problems/decode-ways/description/
# A message containing letters from A-Z can be encoded into numbers using the following mapping:
# 'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"
# Input: s = "226" Output: 3
def decodeWays(s: str) -> int:
    n = len(s)
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1 if s[0] != "0" else 0
    for i in range(2, n + 1):
        # 如果 i-1 = 0 那么 i-1 只能和 i-2 组合
        if s[i - 1] != "0":
            dp[i] += dp[i - 1]
        if "10" <= s[i - 2:i] <= "26":
            dp[i] += dp[i - 2]
    return dp[n]

def decodeWaysConst(s):
    n = len(s)
    # 使用两个变量而非数组来降低空间复杂度
    one_back = 1 if s[0] != "0" else 0  # dp[i-1]
    two_back = 1  # dp[i-2]
    for i in range(2, n + 1):
        current = 0
        # 单独解码的情况
        if s[i - 1] != "0":
            current = one_back
        # 与前一个字符组合解码的情况, 如果是 10，那么current = 0 所以 current = two_back
        if "10" <= s[i - 2:i] <= "26":
            current += two_back
        # 更新状态
        two_back, one_back = one_back, current
    return one_back if n != 0 else 0

#https://leetcode.com/problems/coin-change/description/
# You are given an integer array coins representing coins of different denominations
# and an integer amount representing a total amount of money.
# Return the fewest number of coins that you need to make up that amount.
# If that amount of money cannot be made up by any combination of the coins, return -1.
def coinChange(coins: List[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1

#https://leetcode.com/problems/maximum-product-subarray/description/
# Given an integer array nums, find the contiguous subarray within an array (containing at least one number)
# which has the largest product. nums = [2,3,-2,4] Output: 6
def maxProduct(nums: List[int]) -> int:
    if not nums:
        return 0
    max_val = min_val = res = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_val, min_val = min_val, max_val
        # 假设 num>max_val * num 那么 max_val = num合理 反之依然
        max_val = max(num, max_val * num)
        min_val = min(num, min_val * num)
        res = max(res, max_val)
    return res

#https://leetcode.com/problems/word-break/description/
# Given a string s and a dictionary of strings wordDict,
# return true if s can be segmented into a space-separated sequence of one or more dictionary words.
# 类似coinChange
def wordBreak(s: str, wordDict: List[str]) -> bool:
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[-1]

#https://leetcode.com/problems/longest-increasing-subsequence/description/
# Given an integer array nums, return the length of the longest strictly increasing subsequence.
"""Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
所以并不需要连续
"""
def lengthOfLIS(nums: List[int]) -> int:
    n = len(nums)
    # dp 是 以 nums[i] 结尾的最长子序列
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            # i 比 j 大，increasing seq 就是  nums[i] > nums[j]， 因为不需要连续所以dp[j] + 1成立
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

#https://leetcode.com/problems/partition-equal-subset-sum/description/
# Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets
# such that the sum of elements in both subsets is equal. [1,5,11,5] Output: true [1, 5, 5] and [11]
# 就是 coinChange 问题
def canPartition(nums: List[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]