from typing import Optional


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

#Return the head of the copied linked list.
def copyRandomList(head: Optional[Node]) -> Optional[Node]:
    if not head:
        return None
    # 第一步，复制节点，head->A->B->C 变成 head->A->A'->B->B'->C->C'
    # head 就是 A
    cur = head
    while cur:
        tmp = Node(cur.val, cur.next)
        cur.next = tmp
        cur = tmp.next
    # 第二步，复制random指针 head->A->A' 然后A'.random = A.random.next
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    # 第三步，拆分链表 head->A->A'->B->B'->C->C' 变成 head->A->B->C 和 A'->B'->C'
    cur = res = head.next
    # pre 是原链表的指针，cur 是新链表的指针
    pre = head
    while cur.next:
        pre.next = pre.next.next
        cur.next = cur.next.next
        pre = pre.next
        cur = cur.next
    pre.next = None
    return res

print(f"copyRandomList {copyRandomList(Node(1))}")
