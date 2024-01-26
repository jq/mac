from collections import defaultdict
from typing import List


def maxProfit(prices: List[int]) -> int:
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit

def maxProfit_py(prices: List[int]) -> int:
    return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))


#当 s[right] 即将加入窗口时发现已经存在于 window 集合中，这意味着找到了一个重复字符。此时，需要缩小窗口的左边界直到移除那个重复的字符。
# 但是，这个重复的字符不一定是 s[left]。这里 window.remove(s[left]) 实际上是在逐步移动左边界，每次循环移除一个字符，
# 直到 s[right] 能够加入窗口中而不引起重复。正确的做法应该是在循环中检查并移除左边的字符，直到重复的字符被移除。
# 代码中的处理方式简化了这个过程，每次遇到重复字符时只移动一次左边界，并不立即检查 s[right] 是否还在窗口中重复。
# 这是因为下一次循环会再次尝试将 s[right] 加入窗口，并在必要时继续移动左边界。
def lengthOfLongestSubstring(s: str) -> int:
    left, right = 0, 0
    max_len = 0
    window = set()
    while right < len(s):
        if s[right] not in window:
            window.add(s[right])
            right += 1
            max_len = max(max_len, right - left)
        else:
            window.remove(s[left])
            left += 1
    return max_len

def lengthOfLongestSubstring_dict(s: str) -> int:
    charIndexMap = {}
    left = 0
    max_len = 0
    for right in range(len(s)):
        if s[right] in charIndexMap:
            # 直接跳到重复字符的下一个位置
            left = max(left, charIndexMap[s[right]] + 1)
        # 更新字符的最新索引
        charIndexMap[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len


#
def characterReplacement(s: str, k: int) -> int:
    left, right = 0, 0
    max_len = 0
    # window保存字符出现的次数
    window = defaultdict(int)
    while right < len(s):
        window[s[right]] += 1
        right += 1
        # 如果窗口中的字符数量加上k小于窗口长度，说明窗口无法全部替换成相同字符
        while right - left - max(window.values()) > k:
            # 移动左边界
            window[s[left]] -= 1
            left += 1
        max_len = max(max_len, right - left)
    return max_len

#Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.
# s2的一个和s1 相同长度的window 正好 包含 s1 所有的字符以及相应的数量
def checkInclusion(s1: str, s2: str) -> bool:
    need = defaultdict(int)
    window = defaultdict(int)
    for c in s1:
        need[c] += 1
    left, right = 0, 0
    valid = 0
    while right < len(s2):
        c = s2[right]
        right += 1
        if need[c]:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1
        while right - left >= len(s1):
            if valid == len(need):
                return True
            d = s2[left]
            left += 1
            if need[d]:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    return False

from collections import Counter

def checkInclusion_counter(s1: str, s2: str) -> bool:
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 > len_s2:
        return False

    s1_count = Counter(s1)
    window_count = Counter(s2[:len_s1-1])

    for i in range(len_s1-1, len_s2):
        # Add the new character to the current window
        window_count[s2[i]] += 1
        # If the count of the characters in the current window matches s1, return True
        if window_count == s1_count:
            return True
        # Remove the character that is moving out of the window
        window_count[s2[i-len_s1+1]] -= 1
        # If the count goes to zero, remove it from the counter to prevent size buildup
        if window_count[s2[i-len_s1+1]] == 0:
            del window_count[s2[i-len_s1+1]]
    return False


def minWindow(s: str, t: str) -> str:
    need = Counter(t)
    window = defaultdict(int)
    left, right = 0, 0
    valid = 0
    start, min_length = 0, float('inf')
    while right < len(s):
        c = s[right]
        right += 1
        if need[c]:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1
        while valid == len(need):
            cur_window = right - left
            if cur_window < min_length:
                start = left
                min_length = cur_window
            # 移动左边界,
            d = s[left]
            left += 1
            #左边的被剔除的字符如果在 need 中，需要更新 window 和 valid
            if need[d]:
                if window[d] == need[d]:
                    # 如果是有效字符，需要减少valid，因为之前valid是最小数量，
                    # 这样就会跳出while循环，继续移动右边界
                    valid -= 1
                #无论是不是在need中，都需要更新window
                window[d] -= 1
    return '' if min_length == float('inf') else s[start:start+min_length]


def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    res = []
    # 保存的是索引
    window = []
    for i, num in enumerate(nums):
        if i >= k and window[0] <= i - k:
            window.pop(0)
        while window and nums[window[-1]] <= num:
            window.pop()
        window.append(i)
        if i >= k - 1:
            res.append(nums[window[0]])
    return res

from collections import deque
def maxSlidingWindow_deque(nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    res = []
    # 保存的是索引
    window = deque()
    for i, num in enumerate(nums):
        # window 0 不在窗口内 remove，确保 window是valid的,不会有无效的值
        left = i - k
        if window and window[0] == left:
            window.popleft()
        # 由于我们总是移除所有小于当前考虑的新元素的值的索引，这保证了一旦这个新元素被添加到队列中，它之前的任何元素都不会比它小。
        # 因此，如果这个新元素是当前最大的，它就会添加到队列的尾部。如果不是，那么它之前的元素（在队列中）必定是更大的，
        while window and nums[window[-1]] <= num:
            window.pop()
        window.append(i)
        # i == k - 1 时 是第一个窗口结束
        if i >= k - 1:
            res.append(nums[window[0]])
    return res