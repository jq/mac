from collections import deque
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 递归加速 需要有重复access，没有的话就不需要
def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root


def invertTreeIterative(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    # 可以用 stack = [root]
    queue = deque([root])
    while queue:
        # popleft 是先左后右，pop是先右后左
        # popleft 是广度优先，因为n 层的节点都在n+1层节点前处理
        current_node = queue.popleft()
        # Swap the children
        current_node.left, current_node.right = current_node.right, current_node.left
        # Add the children to the queue if they are not None
        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)

    return root

def maxDepth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

def maxDepthIterative(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    queue = deque([root])
    depth = 0
    while queue:
        depth += 1
        for _ in range(len(queue)):
            current_node = queue.popleft()
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
    return depth

def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    ans = 0
    def depth(node):
        # 当你在一个函数内部定义另一个函数时，内部函数可以访问外部函数的变量。但是，这种访问默认是只读的。
        # 如果你尝试在内部函数中修改外部函数的变量的值，Python会将这视为在内部函数的局部作用域内创建一个新的同名变量，而不是修改外部函数的变量
        nonlocal ans
        if not node: return 0
        L = depth(node.left)
        R = depth(node.right)
        # 所有节点的左右子树的深度之和的最大值
        ans = max(ans, L+R)
        return max(L, R) + 1

    depth(root)
    return ans

# 似乎比递归慢
def diameterOfBinaryTreeIterative(root: Optional[TreeNode]) -> int:
    stack = []
    stack.append(root)
    ans = 0
    depth = {}
    while stack:
        node = stack[-1]
        if node.left and node.left not in depth:
            stack.append(node.left)
        elif node.right and node.right not in depth:
            stack.append(node.right)
        else:
            node = stack.pop()
            left = depth.get(node.left, 0)
            right = depth.get(node.right, 0)
            depth[node] = max(left, right) + 1
            ans = max(ans, left + right)

    return ans

# 平衡二叉树 是一棵空树或它的左右两个子树的高度差的绝对值不超过1
def isBalanced(root: Optional[TreeNode]) -> bool:
    def check(root):
        if not root:
            return 0
        left = check(root.left)
        right = check(root.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return check(root) != -1


def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return isSameTree(p.right, q.right) and isSameTree(p.left, q.left)


def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    if not root:
        return False
    if isSameTree(root, subRoot):
        return True
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)

# 保持p q 然后遍历tree 如果找到
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    #  如果在左子树和右子树中都找到了p和q,那么root就是最近公共祖先，否则，如果左子树找到了p或q，右子树没有找到，那么p或q就是最近公共祖先
    # root 的父亲不会是 在左右都找到， 所以在低处找到的会一直 return 到最高处
    if left and right:
        return root
    return left or right

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    queue = deque([root])
    res = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res

def rightSideView(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    queue = deque([root])
    res = []
    while queue:
        # 只取最右边的，就是level order的最后一个
        res.append(queue[-1].val)
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return res

# if in the path from root to X there are no nodes with a value greater than X.
# 所以root一定是，接下来只要pass max value 就行
def goodNodes(root: TreeNode) -> int:
    def dfs(node, max_val):
        if not node:
            return 0
        # 如果当前节点的值大于等于max_val，那么当前节点就是一个good node
        if node.val >= max_val:
            max_val = node.val
            return 1 + dfs(node.left, max_val) + dfs(node.right, max_val)
        else:
            return dfs(node.left, max_val) + dfs(node.right, max_val)
    return dfs(root, root.val)

def isValidBstRecursive(root: Optional[TreeNode]) -> bool:
    def helper(node, lower=float('-inf'), upper=float('inf')):
        if not node:
            return True
        val = node.val
        if val <= lower or val >= upper:
            return False
        if not helper(node.right, val, upper):
            return False
        if not helper(node.left, lower, val):
            return False
        return True
    return helper(root)


# non recursive 中序遍历
def isValidBST(root: Optional[TreeNode]) -> bool:
    stack, inorder = [], float('-inf')
    while stack or root:
        # 中序遍历，先把左边的进stack
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        # If next element in inorder traversal
        # is smaller than the previous one
        # that's not BST.
        if root.val <= inorder:
            return False
        inorder = root.val
        root = root.right
    return True

# 遍历到list 然后return k位置
def kthSmallestrecursive(root, k):
    def inorder(r):
        if not r:
            return []
        return inorder(r.left) + [r.val] + inorder(r.right)

    return inorder(root)[k - 1]

def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if not k:
            return root.val
        root = root.right


def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0]) # 找到根节点在中序遍历中的位置, 左边是左子树，右边是右子树
    #inorder[:mid] 左， 前序遍历 第一个是根节点，所以是1:mid+1，因为右子树在mid+1
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])
    return root

def buildTreeFast(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    # 将中序遍历的值及其索引存入字典，以加快查找速度
    inorder_index_map = {val: index for index, val in enumerate(inorder)}

    def arrayToTree(left: int, right: int) -> Optional[TreeNode]:
        # 如果没有元素构造二叉树，返回None
        if left > right:
            return None

        # 根节点是前序遍历的第一个元素
        root_val = preorder.pop(0)
        root = TreeNode(root_val)

        # 分割中序遍历，确定左右子树
        index = inorder_index_map[root_val]

        # 递归构建左子树和右子树
        root.left = arrayToTree(left, index - 1)
        root.right = arrayToTree(index + 1, right)

        return root

    return arrayToTree(0, len(inorder) - 1)

def maxPathSum(root: Optional[TreeNode]) -> int:
    max_sum = float('-inf')
    def max_gain(node):
        nonlocal max_sum
        if not node:
            return 0
        # max sum on the left and right sub-trees of node
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        totalpath = node.val + left_gain + right_gain
        # update max_sum if it's better to start a new path
        max_sum = max(max_sum, totalpath)
        # for recursion :
        # 因为要和上一层的节点连接，所以只返回左右子树中的最大值 加上自己
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum