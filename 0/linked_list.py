from typing import Optional, List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __len__(self):
        length = 0
        current = self
        while current:
            length += 1
            current = current.next
        return length
    def printList(self):
        current = self
        while current:
            print(current.val, end=' -> ')
            current = current.next
        print('None')
    def buildList(self, values):
        current = self
        for val in values:
            current.next = ListNode(val)
            current = current.next
        return self
    def assertList(self, expected):
        current = self
        for expected_value in expected_values:
            assert current is not None, "链表比预期短"
            assert current.val == expected_value, f"预期值为 {expected_value}, 但得到 {current.val}"
            current = current.next
        assert current is None, "链表比预期长"

def printList(head):
    current = head
    while current:
        print(current.val, end=' -> ')
        current = current.next
    print('None')


def reverseList_normal(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    curr = head
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    return prev

def reverseList(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev

def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if not list1:
        return list2
    if not list2:
        return list1
    if list1.val < list2.val:
        list1.next = mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoLists(list1, list2.next)
        return list2

def mergeTwoLists_direct(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    while list1 and list2:
        if list1.val < list2.val:
            tail.next, list1 = list1, list1.next
        else:
            tail.next, list2 = list2, list2.next
        tail = tail.next
    # python 的 or 语法，如果list1不为空，就返回list1，否则返回list2 类似其他语音的三目运算符？
    tail.next = list1 or list2
    return dummy.next
def findMiddle(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow

def mergeList(first, second):
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next

def reorderList(head: Optional[ListNode]) -> None:
    if not head:
        return
    middle = findMiddle(head)
    printList(middle)
    second = reverseList(middle)
    mergeList(head, second)
    printList(head)
    return head

# reorderList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))
def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    # why use dummy node?
    # 1. 为了处理删除头节点的情况， 由于哑节点被放置在实际头节点之前，函数结束时返回dummy.next可以无缝返回更新后的列表，
    # 无论头节点是否被移除。这消除了在移除操作后返回正确头节点的额外检查的需要。
    # 2. 使用哑节点通过消除当需要移除的节点位于链表头部时处理特殊情况的需要，简化了代码
    dummy = ListNode(0, head)
    slow = fast = dummy
    # move n steps, 然后slow 和 fast 一起移动，直到fast到达尾部的时候slow就对应n from end.
    for _ in range(n):
        fast = fast.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next
    slow.next = slow.next.next
    return dummy.next

# 2 -> 4 -> 3 + 5 -> 6 -> 4 = 7 -> 0 -> 8 输入是reverse order
def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    carry = 0
    current = dummy
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        # divmod 返回的是商和余数
        carry, val = divmod(val1 + val2 + carry, 10)
        current.next = ListNode(val)
        current = current.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next


l = addTwoNumbers(ListNode(2, ListNode(4, ListNode(3))), ListNode(5, ListNode(6, ListNode(4))))
#printList(l)

# fast 可以走两步，slow 只能走一步，如果有环，fast 和 slow 会相遇
def hasCycle(head: Optional[ListNode]) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False

#重复数字相当于 一个环，把位置作为指针，重复数字相当于有两个指针指向相同的位置，就构成了环
# 循环的入口，就是有两个指针指向的位置，因为数字相当于指针/位置，这就是重复的数字
# 假设环外部分的长度是 L, 从环入口到快慢指针首次相遇点的长度是 x，快慢指针首次相遇时慢指针走过的总距离是 L+x
# 快指针每次移动两步，快指针走过 2（L+x), 假设环长度是 C， 那么快指针 走过的路径是也可以表达成 L+x+nC
# 因为快指针走过慢指针走过的路径 加上在环内重复了 extra 的n 次
# 所以 L+x= nc => L = nC - x
# 从数组起点到环入口的距离 == 从首次相遇点绕环n 次 减去环入口到相遇点的距离。
# 因为现在快指针在相遇点，所以将慢指针移到数组起点，然后两个指针以相同的速度移动，再次相遇的位置就是环的入口
# 长度n+1, 数字范围1到n, 有一个重复数字
def findDuplicate(nums: List[int]) -> int:
    slow = fast = 0
    while True:
        slow, fast = nums[slow], nums[nums[fast]]
        if slow == fast:
            slow = 0
            while slow != fast:
                slow, fast = nums[slow], nums[fast]
            return slow

# 2分法只需要logn次合并
def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    left = mergeKLists(lists[:mid])
    right = mergeKLists(lists[mid:])
    return mergeTwoLists(left, right)


# 每次reverse k 个，直到整个list reverse, K个一组翻转链表
def reverseKGroup(head, k):
    dummy = ListNode(0)
    dummy.next = head
    prev, end = dummy, dummy

    while end.next:
        # 寻找当前段的尾节点
        for _ in range(k):
            end = end.next
            if not end:
                return dummy.next
        start = prev.next  # 当前段的头节点
        next = end.next  # 下一段的起始节点
        end.next = None  # 断开当前段和下一段的连接
        prev.next = reverseList(start)  # 翻转当前段，并连接前一段
        start.next = next  # 头已经变尾部，连接到下一段
        prev = start  # 为下一次迭代更新prev
        end = prev  # 重置end为prev，准备下一次循环
    return dummy.next


reverseKGroup(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))), 2)
reverseKGroup(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))), 3)
