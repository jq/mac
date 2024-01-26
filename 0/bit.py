# https://leetcode.com/problems/single-number/description/
# every element appears twice except for one. Find that single one.
from typing import List
"""Identity: a ^ 0 = a. XORing a number with 0 returns the original number.
Invertibility: a ^ b ^ b = a. XORing a number with another number twice cancels out the second number, returning the original number.
Commutativity: a ^ b = b ^ a. The order of XOR operations doesn't matter.
Associativity: (a ^ b) ^ c = a ^ (b ^ c). The grouping of numbers doesn't affect the result."""
# Input: nums = [2,2,1] Output: 1
def singleNumber(nums: List[int]) -> int:
    res = 0
    for num in nums:
        res ^= num
    return res


#https://leetcode.com/problems/number-of-1-bits/description/
# Write a function that takes an unsigned integer and returns the number of '1' bits it has
# (also known as the Hamming weight).
def hammingWeight2(n: int) -> int:
    res = 0
    while n:
        res += n & 1
        n >>= 1
    return res

def hammingPython(n: int) -> int:
    return bin(n).count("1")
"""如果n是奇数，它的二进制表示的最低位是1。减去1将会改变最低位的1到0（并且不会改变其他位），因此n & (n - 1)的结果就是把n的最低位的1去掉。
如果n是偶数，它的二进制表示的最低位是0。在这种情况下，减1会影响到n中的更高位（从右数第一个1变为0，并且该位之后的所有0都变成1）。
然后，当执行n & (n - 1)时，原来的那个1及其右边的所有0都会被清零。"""
def hammingWeight3(n: int) -> int:
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count

#https://leetcode.com/problems/counting-bits/description/
"""Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10  
3 --> 11  2个1
4 --> 100
5 --> 101
如果x是偶数，那么x中1的数量与x/2中1的数量相同，如果x是奇数，那么x中1的数量比x-1中1的数量多1，
"""
def countBits(n: int) -> List[int]:
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        # 偶数（i & 1 == 0）
        bits[i] = bits[i >> 1] + (i & 1)
    return bits

# https://leetcode.com/problems/reverse-bits/description/
# Reverse bits of a given 32 bits unsigned integer.
def reverseBits(n: int) -> int:
    res = 0
    for i in range(32):
        # 这个操作实际上是把n的最低位复制到res的当前最低位，然后n右移，res左移
        res = (res << 1) + (n & 1)
        n >>= 1
    return res

# https://leetcode.com/problems/missing-number/
# Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
# Input: nums = [3,0,1]
# Output: 2
def missingNumber(nums: List[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum
"""异或操作（XOR，^）有一个特性：一个数与自己异或的结果为0，且异或操作满足交换律和结合律。因此，如果我们将所有的索引和所有的元素进行异或操作，
然后再与n异或（因为缺少的数字可能是n），由于数组中除了缺失的那个数字外，每个数字都出现了两次（一次是数组中的值，一次是索引），"""
def missingNumberBit(nums: List[int]) -> int:
    res = 0
    for i, num in enumerate(nums):
        res ^= i ^ num
    return res

#https://leetcode.com/problems/sum-of-two-integers/description/
# Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.
"""不考虑进位的加法：两个二进制位相加，如果是1 + 1或0 + 0，结果是0；如果是1 + 0或0 + 1，结果是1。这和异或操作的结果一样。
计算进位：只有当两个位都是1时，才会产生进位，所以可以用与操作(&)计算进位，然后将结果左移一位，因为进位会影响到下一位。
重复以上两步：将不考虑进位的加法结果和进位结果再次进行这两个操作，直到没有进位为止。
负数处理：如果最终结果是负数，它会超过32位整数的范围。使用~(a ^ mask)转换回正确的负数结果。"""
def getSum(a: int, b: int) -> int:
    mask = 0xFFFFFFFF
    while b:
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)

#https://leetcode.com/problems/reverse-integer/description/
# Given a signed 32-bit integer x, return x with its digits reversed.
def reverse(x: int) -> int:
    INT_MIN, INT_MAX = -2**31, 2**31 - 1
    res = 0
    negative = x < 0
    x = abs(x)

    while x:
        digit = x % 10
        x //= 10
        # 检查溢出
        if res > INT_MAX // 10 or (res == INT_MAX // 10 and digit > INT_MAX % 10):
            return 0
        res = res * 10 + digit

    if negative:
        res = -res

    return res
