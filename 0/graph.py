#https://leetcode.com/problems/number-of-islands/description/
from collections import deque
from typing import List

#'1's (land) and '0's (water)
def numIslands(grid: List[List[str]]) -> int:
    if not grid:
        return 0

    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            dfs(i + x, j + y)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count

#https://leetcode.com/problems/max-area-of-island/description/
def maxAreaOfIsland(grid: List[List[int]]) -> int:
    if not grid:
        return 0

    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return 0
        grid[i][j] = 0
        return 1 + sum(dfs(i + x, j + y) for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)])

    return max(dfs(i, j) for i in range(len(grid)) for j in range(len(grid[0])))

#https://leetcode.com/problems/pacific-atlantic-water-flow/description/
def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    if not heights:
        return []

    m, n = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(i, j, ocean):
        ocean.add((i, j))
        for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            ni, nj = i + x, j + y
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in ocean and heights[ni][nj] >= heights[i][j]:
                dfs(ni, nj, ocean)
# 是看从边界开始，能不能流到里面, 所以不用检查所有点，就是从4条边开始检查
    for i in range(m):
        dfs(i, 0, pacific)
        dfs(i, n - 1, atlantic)
    for j in range(n):
        dfs(0, j, pacific)
        dfs(m - 1, j, atlantic)

    return list(pacific & atlantic)

# https://leetcode.com/problems/surrounded-regions/description/
# 从边界开始，把O 都mark成A，然后再把A变成O，其他的变成X
def surrounded_region(board: List[List[str]]) -> None:
    if not board:
        return

    m, n = len(board), len(board[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
            return
        board[i][j] = 'A'
        for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            dfs(i + x, j + y)

    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)

    for i in range(m):
        for j in range(n):
            if board[i][j] == 'A':
                board[i][j] = 'O'
            else:
                board[i][j] = 'X'

#https://leetcode.com/problems/rotting-oranges/description/
def orangesRotting(grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Initialize the queue with all rotten oranges and count fresh oranges
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                fresh += 1
            elif grid[i][j] == 2:
                queue.append((i, j))

    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    minutes = 0

    # BFS to spread the rot
    while queue and fresh > 0:
        for _ in range(len(queue)):
            i, j = queue.popleft()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    queue.append((ni, nj))
        minutes += 1

    return minutes if fresh == 0 else -1

#https://www.lintcode.com/problem/663/
"""-1 - A wall or an obstacle.
0 - A gate.
INF - Infinity means an empty room.
Fill each empty room with the distance to its nearest gate. 
If it is impossible to reach a Gate, that room should remain filled with INF
基本思路是从每个门（值为0的单元格）开始进行搜索，逐层向外扩展，直到填充所有可达的房间。对于每个房间，我们使用其到最近门的距离来更新其值。
如果一个房间无法到达任何门，则保持其值为无穷大（INF）。"""
def walls_and_gates(rooms: List[List[int]]):
    if not rooms:
        return
    rows, cols = len(rooms), len(rooms[0])
    INF = 2147483647
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 上下左右四个方向
    queue = deque()

    # 首先找到所有的门并将它们的位置加入队列，同时从所有的门开始搜索的，而不是依次从每个门开始 所以不用担心覆盖
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))

    # 广度优先搜索，所以只要更新长度就行。不用担心更长的路径覆盖更短
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 如果新位置在范围内，且是一个尚未访问的房间
            if 0 <= nx < rows and 0 <= ny < cols and rooms[nx][ny] == INF:
                rooms[nx][ny] = rooms[x][y] + 1  # 更新距离
                queue.append((nx, ny))  # 将新位置加入队列继续搜索


#https://leetcode.com/problems/course-schedule/
"""Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. 
So it is impossible. 检查课程之间的依赖关系图是否存在环来解决
"""
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = [[] for _ in range(numCourses)]
    visited = [0] * numCourses
    #使用 graph 来构建课程的依赖图，其中 graph[x] 包含了所有课程 x 的先修课程。
    for x, y in prerequisites:
        graph[x].append(y)

    def dfs(i):
        # -1 表示正在访问，1 表示已经访问过 发现环
        if visited[i] == -1:
            return False
        if visited[i] == 1:
            return True
        visited[i] = -1
        for j in graph[i]:
            if not dfs(j):
                return False
        # 该课程所有依赖检查完，不用再检查
        visited[i] = 1
        return True

    for i in range(numCourses):
        if not dfs(i):
            return False
    return True
# https://leetcode.com/problems/course-schedule-ii/
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    graph = [[] for _ in range(numCourses)]
    visited = [0] * numCourses
    res = []

    for x, y in prerequisites:
        graph[x].append(y)

    def dfs(i):
        if visited[i] == -1:
            return False
        if visited[i] == 1:
            return True
        visited[i] = -1
        for j in graph[i]:
            if not dfs(j):
                return False
        visited[i] = 1
        # 前序都加了，
        res.append(i)
        return True

    for i in range(numCourses):
        if not dfs(i):
            return []
    return res[::-1]

# https://www.lintcode.com/problem/3651/
"""In this problem, there is an undirected graph with n nodes. There is also an edges array. Where edges[i] = [a, b] means that there is an edge between node a and node b in the graph.
"""
def count_components(n: int, edges: List[List[int]]) -> int:
    def dfs(node):
        if visited[node]:
            return
        visited[node] = True
        for neighbor in graph[node]:
            dfs(neighbor)

    # Initialize the graph and visited list
    graph = {i: [] for i in range(n)}
    visited = [False] * n

    # Build the graph
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    # Count components
    components = 0
    for i in range(n):
        if not visited[i]:
            dfs(i)
            components += 1
    return components
#https://leetcode.com/problems/redundant-connection/
"""Return an edge that can be removed so that the resulting graph is a tree of n nodes. 
If there are multiple answers, return the answer that occurs last in the input.
肯定有一个边
并查集(Union-Find)算法 用于处理不交集（Disjoint Sets）合并及查询问题的数据结构
利用并查集来帮助我们找到形成环的那条边，这条边就是我们要找的冗余连接
对于每一条边，使用并查集来检查这两个节点是否已经在同一个集合中：
如果在同一个集合中，说明添加这条边会形成一个环，这就是我们要找的冗余边。
如果不在同一个集合中，则将这两个节点合并到同一个集合中。
在并查集中，每个元素都有一个指向其父元素的链接，如果元素是组的根（代表），则其父元素指向它自己
"""
class UnionFind:
    def __init__(self, size):
        # 开始每个节点都是指向自己
        self.parent = list(range(size))

# find 方法的目的是找到给定元素所在的集合的代表（即根元素）
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

# x 是 y 的父节点
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX
            return False
        # rootX == rootY表示两个节点X和Y已经在同一个连通分量中，即它们已经通过某条路径连接起来了。
        # 当你尝试将这两个节点通过一条新的边连接时，如果发现它们已经在同一个连通分量中（即它们的根节点相同）
        # ，这意味着添加这条边会形成一个闭环，因为这条边提供了另一条路径将这两个节点连接起来，从而形成了至少一个环。
        return True

def findRedundantConnection(edges: List[List[int]]) -> List[int]:
    uf = UnionFind(len(edges) + 1)  # 节点编号从1开始，所以要+1
    for x, y in edges:
        if uf.union(x, y):
            return [x, y]
# https://leetcode.com/problems/word-ladder/
"""Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
循环处理队列中的元素，对于每个元素，尝试更改单词的每一个字母为26个英文字母中的任意一个，生成一个新的单词。
如果这个新单词是endWord，那么找到了最短路径，返回当前路径长度。
如果这个新单词在wordSet中，那么将其添加到队列中，并从wordSet中移除该单词（这样做是为了防止重复访问同一个单词）。"""
def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0

    queue = deque([(beginWord, 1)])
    while queue:
        word, length = queue.popleft()
        if word == endWord:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i + 1:]
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    queue.append((next_word, length + 1))
    return 0

#https://www.lintcode.com/problem/178/
"""Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes)
, write a function to check whether these edges make up a valid tree.
图中是否存在环，以及图是否是连通的"""
def valid_tree(n: int, edges: List[List[int]]) -> bool:
    if len(edges) != n - 1:  # 检查边的数量是否正确
        return False
    uf = UnionFind(n)
    for x, y in edges:
        if uf.union(x, y):  # 如果不能合并，说明存在环
            return False
    return True


"""Given a directed graph, design an algorithm to find out whether there is a route between two nodes.
"""
def hasRoute(n: int, edges: List[List[int]], x: int, y: int) -> bool:
    graph = {i: [] for i in range(n)}
    visited = [False] * n

    for a, b in edges:
        graph[a].append(b)

    def dfs(node):
        if node == y:
            return True
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor] and dfs(neighbor):
                return True
        return False

    return dfs(x)

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
#https://leetcode.com/problems/clone-graph/
from typing import Optional

def cloneGraph(node: Optional[Node]) -> Optional[Node]:
    if not node: return node
    # 字典用于存储已经访问过的节点映射
    visited = {}

    # 初始化队列和第一个克隆节点
    queue = deque([node])
    visited[node] = Node(node.val, [])

    while queue:
        current = queue.popleft()
        for neighbor in current.neighbors:
            if neighbor not in visited:
                # 如果邻居节点未访问，创建克隆并加入队列
                visited[neighbor] = Node(neighbor.val, [])
                queue.append(neighbor)
            # 更新当前克隆节点的邻居列表
            visited[current].neighbors.append(visited[neighbor])

    return visited[node]