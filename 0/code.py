# https://neetcode.io/roadmap
import operator
from collections import defaultdict
from typing import List


#https://leetcode.com/problems/contains-duplicate/solutions/4759657/video-step-by-step-visualization-of-o-n-solution/
# https://leetcode.com/problems/contains-duplicate/description/
def contain_dup(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))

#https://leetcode.com/problems/contains-duplicate-ii/
def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    d = defaultdict(list)
    for idx, num in enumerate(nums):
        d[num].append(idx)
    for l in d.values():
        for x in range(1, len(l)):
            diff = l[x] - l[x-1]
            if k >= diff:
                return True
    return False

def containsNearbyDuplicate_slidding(nums: list[int], k: int) -> bool:
    seen = set()
    for i, num in enumerate(nums):
        if num in seen:
            return True
        seen.add(num)
        if len(seen) > k:
            seen.remove(nums[i - k])
    return False

# hash table + bucketing
#https://leetcode.com/problems/contains-duplicate-iii/description/
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    if valueDiff < 0 or indexDiff < 0:
        return False  # 边界情况：负的差值在这个上下文中没有逻辑意义。
    buckets = {}
    bucket_width = valueDiff + 1  # 调整桶宽度，确保覆盖valueDiff。
    for i, num in enumerate(nums):
        bucket_id = num // bucket_width  # 确定当前数字的桶。
        # 检查当前桶或邻近桶中的近似重复项。
        if bucket_id in buckets:
            return True
        if (bucket_id - 1 in buckets and abs(buckets[bucket_id - 1] - num) < bucket_width) or \
                (bucket_id + 1 in buckets and abs(buckets[bucket_id + 1] - num) < bucket_width):
            return True
        # 将当前数字加入其对应的桶中。
        buckets[bucket_id] = num
        # 移除桶中已经过时的元素。
        if i >= indexDiff:
            del buckets[nums[i - indexDiff] // bucket_width]
    return False

#https://leetcode.com/problems/valid-anagram/description/
def isAnagram(s: str, t: str) -> bool:
    #sorted(s) == sorted(t)
    from collections import Counter
    return Counter(s) == Counter(t)
#https://leetcode.com/problems/group-anagrams/description/
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    d = defaultdict(list)
    for s in strs:
        # why tuple? because list is not hashable,
        # why list is not hashable? because list is mutable,
        d[tuple(sorted(s))].append(s)
    return list(d.values())

# https://leetcode.ca/all/249.html
'''Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd"
"abc" -> "bcd" -> ... -> "xyz"
Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
Output:
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]'''
def groupStrings(strings: list[str]) -> list[list[str]]:
    d = defaultdict(list)
    for s in strings:
        # 相对于第一个字母的位移
        key = tuple((ord(c) - ord(s[0])) % 26 for c in s)
        d[key].append(s)
    return list(d.values())

def p3(nums, target) -> list[int]:
    d = {}
    for i, num in enumerate(nums):
        pv = target - num
        if pv in d:
            return [d[pv], i]
        d[num] = i
    return []

# Your solution must use only constant extra space.
def twoSum(numbers: List[int], target: int) -> List[int]:
    left, right = 0, len(numbers) - 1
    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]
        elif s < target:
            left += 1
        else:
            right -= 1

def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res = []
    for i, a in enumerate(nums):
        if i > 0 and a == nums[i-1]:
            print(f"skip dup i: {i}, a: {a}, nums[i-1]: {nums[i-1]}")
            #如果和前面元素一样，就已经被计算过了，所以跳过
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            three_sum = a + nums[l] + nums[r]
            if three_sum > 0:
                r -= 1
            elif three_sum < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                # 如果和前面元素一样，就已经被计算过了，所以跳过
                while nums[l] == nums[l-1] and l < r:
                    l += 1
    return res

#print(threeSum([0,0,0]))

def topkFreq(nums: list[int], k: int) -> list[int]:
    from collections import Counter
    import heapq
    count = Counter(nums)
    # count.keys() 是找到的所有int
    # count.get() 找到freq, 也就是按freq排序来找topk 是找到的所有int的value
    return heapq.nlargest(k, count.keys(), key=count.get)

def productExceptSelf(nums: list[int]) -> list[int]:
    res = [1] * len(nums)
    left = right = 1
    for i in range(len(nums)):
        res[i] *= left
        right_index = -i-1
        # what's ~i? it's a bitwise not, ~i = -i-1
        # res[~i] is the same as res[-i-1] 其实是右边的对称index
        res[right_index] *= right
        left *= nums[i]
        right *= nums[right_index]
    return res

def isValidSudoku(board: list[list[str]]) -> bool:
    seen = set()
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = board[i][j]
                # // 是整除
                if (i, num) in seen or (num, j) in seen or (i//3, j//3, num) in seen:
                    return False
                seen.add((i, num))
                seen.add((num, j))
                seen.add((i//3, j//3, num))
    return True

# decode encode string
#define a delimiter which could be any special character like # or / to separate words in encode
# method. However, this special character might appear in the word in real world.
# We can get around it by appending an escaped character or size of the word.
# Here, we append the size of the word when encoding
def encode(self, strs: List[str]) -> str:
    delimiter = "#"
    encoded = delimiter.join(f"{len(s)}:{s}" for s in strs)
    return encoded
def decode(self, s: str) -> List[str]:
    strings = []
    i = 0
    while i < len(s):
        j = s.find(':', i)
        size = int(s[i:j])
        strings.append(s[j+1:j+1+size])
        i = j + 1 + size + 1  # Skip the size, the actual string, and the delimiter
    return strings

def longestConsecutive(nums):
    nums = set(nums)
    best = 0
    for x in nums:
        # 始终从最大的开始找，只需要单边找
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)
    return best

def evalRPN_op(tokens):
    stack = []
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda a, b: int(a/b),  # Use truediv for division, then convert result to int
    }
    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack.pop()
def valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {
        '(': ')',
        '{': '}',
        '[': ']'
    }
    for char in s:
        if char in mapping:
            stack.append(char)
        elif not stack or char != mapping[stack.pop()]:
            return False
    return not stack

def evalRPN(tokens: List[str]) -> int:
    stack = []
    for t in tokens:
        if t not in "+-*/":
            stack.append(int(t))
        else:
            b, a = stack.pop(), stack.pop()
            if t == '+':
                stack.append(a + b)
            elif t == '-':
                stack.append(a - b)
            elif t == '*':
                stack.append(a * b)
            else:
                # 对于正数 a//b == int(a/b), 负数则不相同。应为//是向下取整
                stack.append(int(a / b))
    return stack.pop()

def generateParenthesis(self, n: int) -> List[str]:
    def generate(p, left, right, parens=[]):
        if left:
            generate(p + '(', left-1, right)
        if right > left:
            generate(p + ')', left, right-1)
        if not right:
            parens.append(p)
        return parens
    return generate('', n, n)

def generateParenthesis_1(n: int) -> List[str]:
    def generate(p, left, right):
        print(p, left, right)
        if left == 0 and right == 0:
            return [p]
        parens = []
        if left > 0:
            parens += generate(p + '(', left - 1, right)

        if right > left:
            parens += generate(p + ')', left, right - 1)
        print(left, right, parens)
        return parens

    return generate('', n, n)

# https://chat.openai.com/c/aee482f6-6879-40ec-bdd1-8cfdcc46d0fc
# 递归程序加 level，print的时候显示，在递归函数的开始和结束处打印状态信息，
def generateParenthesis_2(n: int) -> List[str]:
    def generate(p, left, right, depth=0):
        # 使用缩进来可视化递归深度
        indent = ' ' * depth
        if left == 0 and right == 0:
            print(f"{indent}到达叶子: {p}")
            return [p]
        parens = []
        if left > 0:
            print(f"{indent}添加左括号: {p}( {left} {right} {depth}")
            out = generate(p + '(', left - 1, right, depth + 1)
            parens += out
            print(f"{indent}左括号递归结束: {out} {left} {right} {depth}")
        if right > left:
            print(f"{indent}添加右括号: {p}) {left} {right} {depth}")
            out = generate(p + ')', left, right - 1, depth + 1)
            parens += out
            print(f"{indent}右括号递归结束: {out} {left} {right} {depth}")
        return parens

    return generate('', n, n)
#generateParenthesis_2(3)

# Input: temperatures = [73,74,75,71,69,72,76,73]
# Output: [1,1,4,2,1,1,0,0]
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    # stack stores index of temperatures lower or equal to current temperature
    # 如果当前温度比栈顶温度高，就挨个pop出来，然后计算index差
    # 反之，就push进去，等到下一个高温
    # 需要用stack 的原因是 有可能遇到 75,71,69,72， 那么72可以clear 71 69,但需要保留 75 等下一个高温
    stack = []
    res = [0] * len(temperatures)
    for i, t in enumerate(temperatures):
        print(f"i: {i}, t: {t}, stack: {stack}, res: {res}")
        while stack and t > temperatures[stack[-1]]:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res

def carFleet(target: int, position: List[int], speed: List[int]) -> int:
    pair = sorted(zip(position, speed), reverse=True)
    print(f"pair: {pair}")
    stack = []
    for p, s in pair:
        time = (target - p) / s
        print(f"time: {time}")
        # 如果当前车的时间比栈顶车的时间短，就说明栈顶车会被追上，然后当前车也没法超车，所以不用push
        if not stack or time > stack[-1]:
            stack.append(time)
            print(f"stack: {stack}")
    return len(stack)

#carFleet(12, [10,8,0,5,3], [2,4,1,1,3])
def largestRectangleArea(heights: List[int]) -> int:
    stack = []
    max_area = 0
    # 这里为什么要加一个0，是因为如果不加，最后一个柱子的高度就不会被计算到
    for i, h in enumerate(heights + [0]):
        # 如果当前高度比栈顶高度小，就意味着之前高度的矩形不能再扩展了
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            # 如果栈为空，说明当前弹出的柱子高度是目前为止遇到的最低高度, 所以宽度就是i
            # 原因是 如果栈不为空，说明栈顶的柱子高度大于之前进stack的柱子高度,
            # 宽度就是i - stack[-1] - 1 的原因是 现在要计算的是从 i-1 的位置到 stack[-1] 的位置的距离
            # 因为前面已经pop了，所以stack[-1]最大的情况就是 i-2
            if not stack:
                width = i
            else:
                width = i - stack[-1] - 1
                print(f"i: {i}, h: {h}, stack: {stack} w: {width}")
            max_area = max(max_area, height * width)
        stack.append(i)
        print(f"push i: {i}, h: {h}, stack: {stack}, max_area: {max_area}")
    return max_area

#largestRectangleArea([2,1,5,6,2,3])

def isPalindrome(s: str) -> bool:
    s = ''.join(filter(str.isalnum, s)).lower()
    return s == s[::-1]

def isPalindrome2(s: str) -> bool:
    s = [c.lower() for c in s if c.isalnum()]
    return all (s[i] == s[~i] for i in range(len(s)//2))

def maxArea(height: List[int]) -> int:
    max_area = 0
    left, right = 0, len(height) - 1
    while left < right:
        max_area = max(max_area, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area

def trap(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = ans = 0
    while left < right:
        if height[left] < height[right]:
            # 左边低 就看左边的最大值
            left_max = max(left_max, height[left])
            # 比左边最大值小，就一定有水
            water = left_max - height[left]
            if water > 0:
                print(f" w {water} left_max: {left_max}, left: {left}, h {height[left]} | right: {right}, h : {height[right]}")
            ans += water
            left += 1
        else:
            right_max = max(right_max, height[right])
            water = right_max - height[right]
            ans += right_max - height[right]
            if water > 0:
                print(f" w {water} right_max: {right_max}, right: {right}, h {height[right]} | left: {left}, h : {height[left]}")
            right -= 1
    return ans

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))