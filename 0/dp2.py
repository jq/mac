# https://leetcode.com/problems/unique-paths/description/
# Given the two integers m and n, 长宽 return the number of possible unique paths
# that the robot can take to reach the bottom-right corner.
import math
from typing import List


def uniquePaths(m: int, n: int) -> int:
    dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]

#https://leetcode.com/problems/longest-common-subsequence/description/
"""Input: text1 = "abcde", text2 = "ace"  和substring不一样，可以不连续 要保持顺序
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
如果两个字符串的头字符相同，那么这个字符必然属于最长公共子序列，我们可以将问题简化为找两个较短字符串（即原字符串各自去掉头字符）的最长公共子序列的长度加一。
如果两个字符串的头字符不同，那么最长公共子序列要么在去掉第一个字符串的头字符后的两个字符串中找到，
要么在去掉第二个字符串的头字符后的两个字符串中找到。
因此，我们取这两种情况的最长公共子序列的较大值。"""

def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/
"""Input: prices = [1,2,3,0,2]
Output: 3
Explanation: transactions = [buy, sell, cooldown, buy, sell]
"""
def maxProfit(prices: List[int]) -> int:
    if not prices:
        return 0

    n = len(prices)
    hold = [0] * n
    cooldown = [0] * n
    rest = [0] * n

    hold[0] = -prices[0]  # 第一天买入股票
    cooldown[0] = 0
    rest[0] = 0

    for i in range(1, n):
        hold[i] = max(hold[i-1], rest[i-1] - prices[i])  # 保持持有或者从rest状态买入
        cooldown[i] = hold[i-1] + prices[i]  # 前一天持有，今天卖出
        rest[i] = max(rest[i-1], cooldown[i-1])  # 保持rest或者从cooldown转换来

    return max(cooldown[n-1], rest[n-1])  # 最后一天不持有股票的状态

#https://leetcode.com/problems/coin-change-ii/description/
"""Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1"""
def change(amount: int, coins: List[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1  # 金额为0的方法数为1，即不使用任何硬币

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]

    return dp[amount]

#https://leetcode.com/problems/target-sum/description/
"""Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3"""
def findTargetSumWays(nums: List[int], target: int) -> int:
    total_sum = sum(nums)
    # 如果总和小于目标的绝对值，或者(total_sum + target)不是偶数，则无法分割为两个子集
    # sum(A) - sum(B) = target  sum(A) + sum(B) + target = 2 A
    if total_sum < abs(target) or (total_sum + target) % 2 == 1:
        return 0

    s = (total_sum + target) // 2
    # 找到 找出数组中元素之和为(sum + target) / 2的子集数量。
    # C(n, k) = n！/(k! * (n-k))! 因为数字不是都是1，所以不能用这个公式math.factorial
    dp = [0] * (s + 1)
    dp[0] = 1  # 0总和可以通过不选择任何数字来达成

    for num in nums:
        # 逆向更新是保证每种硬币只被计算一次， 第一次用num1，只有 dp[num1] = 1被更新。如果正向更新，num1会被使用多次。
        for i in range(s, num - 1, -1):
            # 逆向更新确保在处理dp[i]时，所有依赖的dp[i - num]值都是基于前一轮次的结果，而不是当前轮次可能已经被当前硬币更新的结果
            # 小的是上次更新的结果。
            dp[i] += dp[i - num]

    return dp[s]

# https://leetcode.com/problems/interleaving-string/description/
# Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
"""想象你在填充一个表格，其中行代表s1的前缀，列代表s2的前缀，而每个单元格的值代表考虑到当前位置，是否能够通过交错s1和s2的相应前缀来形成s3的对应前缀。我们从表格的左上角开始，向右下角移动。

对于表格中的任何位置(i, j)（表示s1的前i个字符和s2的前j个字符），如果这个位置的值是true，那么表示存在一种方法可以交错这些字符以形成s3的前i+j个字符。

动态规划的转移方程如下：

如果当前字符在s3中与s1的当前字符匹配，并且表格的上一个位置也是true（即dp[i-1][j]），则dp[i][j]可以是true。
如果当前字符在s3中与s2的当前字符匹配，并且表格的左边位置也是true（即dp[i][j-1]），则dp[i][j]同样可以是true。"""
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    l1, l2, l3 = len(s1), len(s2), len(s3)
    if l1 + l2 != l3:
        return False

    dp = [[False] * (l2 + 1) for _ in range(l1 + 1)]
    dp[0][0] = True

    for i in range(1, l1 + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, l2 + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])

    return dp[l1][l2]

#https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
# Given an m x n integers matrix, return the length of the longest increasing path in matrix.
"""直观上，我们从矩阵的每个点出发，向四个方向（上、下、左、右）探索，寻找递增路径。我们可以使用DFS逐个访问每个元素，
如果下一个元素比当前元素大，我们就继续递归搜索。为了提高效率，我们使用一个与输入矩阵同等大小的缓存（记忆化矩阵），
来存储从每个点出发的最长递增路径的长度。当我们再次访问到某个点时，
如果它已经被计算过，我们就可以直接使用缓存的结果，而不是重新计算。"""
def longestIncreasingPath(matrix: List[List[int]]) -> int:
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    cache = [[0] * n for _ in range(m)]  # 缓存每个位置的最长递增路径长度

    def dfs(i, j):
        if cache[i][j] != 0:  # 如果这个位置已经被计算过，则直接返回结果
            return cache[i][j]

        # 四个方向上的递增路径长度
        path_length = 1  # 至少包括自己，所以起始长度为1
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            # 条件matrix[x][y] > matrix[i][j]，它确保了递归的方向总是向值更大的单元格进行。 所以不用visiting
            if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                path_length = max(path_length, 1 + dfs(x, y))

        cache[i][j] = path_length  # 更新缓存
        return path_length

    return max(dfs(i, j) for i in range(m) for j in range(n))

# https://leetcode.com/problems/distinct-subsequences/description/
"""return the number of distinct subsequences of s which equals t.Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from s.
如果s的当前字符与t的当前字符匹配，我们可以选择使用或不使用这个字符。如果我们选择使用它，
就需要在s的剩余部分中找到t的剩余部分。如果我们选择不使用它，我们仍然需要在s的剩余部分中找到整个t。"""
def numDistinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    #dp[i][j]表示s的前i个字符中包含t的前j个字符的不同子序列的数量
    # 当t为空字符串时，s的任何前缀都只包含一个空字符串作为子序列
    for i in range(m + 1):
        dp[i][0] = 1

    for i in range(1, m + 1):
        # 当s的当前考虑长度i小于t的总长度n时，s无法包含完整的t
        # 只有当j <= i时，子序列计数才有意义，因为我们只能在s的前i个字符中寻找t的前j个字符。
        # 不直接使用i 是因为 i > n 的时候 最多长度就是n
        for j in range(1, min(i, n) + 1):
            if s[i - 1] == t[j - 1]:
                # dp[i - 1][j - 1] 在s的前i-1个字符中找到t的前j-1个字符的不同子序列的数量，代表使用这个匹配
                # dp[i - 1][j] 代表不使用这个匹配，就是还要用i-1 和 J 来匹配。
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[m][n]

# https://leetcode.com/problems/edit-distance/description/
"""如果当前两个字符不同，我们需要考虑三种操作：
插入：在word1中插入一个字符后，问题转化为比较新的word1和去掉一个字符的word2。
删除：从word1中删除一个字符，问题转化为比较去掉一个字符的word1和原始的word2。
替换：将word1的当前字符替换为word2的当前字符，问题转化为比较这两个字符串去掉这个字符的剩余部分。"""
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 动态规划填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需编辑
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # 删除
                                   dp[i][j - 1],    # 插入
                                   dp[i - 1][j - 1]) # 替换
    return dp[m][n]

# https://leetcode.com/problems/burst-balloons/description/
"""Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
最后一个爆破的气球是k，那么在爆破气球k之前，气球k的左边和右边的气球就不会相邻。
因此，我们可以把问题分解为两部分：爆破气球k左边的所有气球获得的最大硬币数，
加上爆破气球k右边的所有气球获得的最大硬币数，
再加上最后爆破气球k获得的硬币数。这样，原问题就被分解为了规模更小的相同问题
"""
def maxCoins(nums: List[int]) -> int:
    # 在两端添加1，方便处理边界情况
    nums = [1] + nums + [1]
    n = len(nums)
    # dp[i][j]表示区间(i, j)能获得的最多硬币数
    dp = [[0] * n for _ in range(n)]

    # k是区间长度，i，m,j 包括一个气球的最小集合长度是2，
    for k in range(2, n):
        # i是区间起始点
        for i in range(n - k):
            j = i + k
            # 在区间(i, j)中尝试每个点作为最后一个被爆破的气球，
            # 因为m是需要被爆破的气球的索引，它必须位于i和j之间，但不能等于这两个边界值，所以循环是range(i+1, j)。
            for m in range(i + 1, j):
                # 选择一个使得值最大的m
                dp[i][j] = max(dp[i][j], nums[i] * nums[m] * nums[j] + dp[i][m] + dp[m][j])

    return dp[0][n - 1]

#https://leetcode.com/problems/regular-expression-matching/description/
"""'.' Matches any single character.
'*' Matches zero or more of the preceding element.  """
def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # 初始化：两个空字符串是匹配的 dp[i][j]表示s的前i个字符和p的前j个字符是否匹配
    dp[0][0] = True

    # 处理p的前j个字符和空字符串s的匹配情况，
    for j in range(2, n + 1):
        # [j - 2] 确保 a*b* 的时候 dp[0][2] = True dp[0][4] = True
        dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] in {s[i - 1], '.'}:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and p[j - 2] in {s[i - 1], '.'})

    return dp[m][n]
