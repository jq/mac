# https://leetcode.com/problems/reconstruct-itinerary/description/
#  all tickets form at least one valid itinerary. You must use all the tickets once and only once.
# 有两个选择的时候选择字典序小的
import heapq
from collections import defaultdict
from typing import List


def findItinerary(tickets: List[List[str]]) -> List[str]:
    graph = defaultdict(list)
    # 逆序存储，保证字典序大的城市被保存在[] 前面，然后dfs中 因为是 pop 所以是 先被access, 这是第一个翻转
    for start, end in sorted(tickets, reverse=True):
        print(f"start: {start}, end: {end}")
        graph[start].append(end)

    result = []
    def dfs(city):
        while graph[city]:
            path = graph[city].pop()
            print(f"pop path: {path}")
            dfs(path)
        print(f"append city: {city}")
        # dfs 是第二个翻转
        result.append(city)
    # 以jfk 为根，dfs 里面是最后打印，all tickets form at least one valid itinerary. 确保一个dfs 可以走完所有的tickets
    dfs("JFK")
    return result[::-1]

#tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"], ["ATL", "SFO"]]
#print(findItinerary(tickets))  # ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]

#https://leetcode.com/problems/min-cost-to-connect-all-points/description/
# Given an array points where points[i] = [xi, yi] represents a point on the X-Y plane,
# return the minimum cost to make all points connected.  manhattan distance |xi - xj| + |yi - yj|,
# 最小生成树 min spanner tree  最小生成树是一种找到能连接图中所有顶点且边的权重总和最小的树。
# 从一个顶点开始，逐步增加边和顶点，直到包含所有顶点，每次都添加连接已选顶点和未选顶点且权重最小的边。
"""对于稠密图（即边的数量接近节点数的平方），Prim算法通常更优，因为Kruskal算法需要处理所有边。对于稀疏图，Kruskal算法可能更有效率，
因为边的数量远小于节点数的平方。Kruskal算法的一个关键优势是其简单性，它不需要一个复杂的数据结构来存储图
本题是一个稠密图，因此我们选择使用Prim算法。从点出发不是从边出发。
"""
def minCostConnectPoints(points: List[List[int]]) -> int:
    # 计算两点间的曼哈顿距离
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    # 初始化一个优先队列
    minHeap = [(0, 0)]  # (cost, point_index)
    visited = set()
    result = 0

    while len(visited) < len(points):
        # 选取权重最小的边
        cost, i = heapq.heappop(minHeap)
        if i in visited:
            continue
        result += cost
        visited.add(i)
        # 未选顶点 按权重构建最小堆
        for j in range(len(points)):
            if j not in visited:
                heapq.heappush(minHeap, (manhattan(points[i], points[j]), j))

    return result

#https://leetcode.com/problems/network-delay-time/description/
# There are N network nodes, labelled 1 to N. Given times, a list of travel times as directed edges times[i] = (u, v, w),
# where u is the source node, v is the target node, and w is the time it takes for a signal to travel from source to target.
# Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal?
# If it is impossible, return -1.
"""最短路径问题，特别是关于如何计算从一个给定的起点到图中所有其他节点的最短时间。这个问题可以通过Dijkstra算法或Bellman-Ford算法来解决。
在这个场景中，我们考虑使用Dijkstra算法，因为它适用于没有负权边的图，我们需要找到从K到图中每个其他节点的最短时间，并返回这些时间中的最大值。
以起点K为中心，逐步扩展到达其他节点的最短路径
times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2  k是开始节点 n是总节点数量  2,1,1 表示从2到1的时间是1
Output: 2"""
def networkDelayTime(times: List[List[int]], n: int, k: int) -> int:
    # 构建图，使用邻接表表示
    graph = {i: [] for i in range(1, n+1)}
    for u, v, w in times:
        graph[u].append((v, w))

    # 初始化距离数组，所有节点距离为无穷大，除了起点
    distance = {i: float('inf') for i in range(1, n+1)}
    distance[k] = 0

    # 使用优先队列优化Dijkstra算法
    minHeap = [(0, k)]
    while minHeap:
        curDist, u = heapq.heappop(minHeap)
        # 无需更新 因为已经是最小值
        if curDist > distance[u]:
            continue
        for v, w in graph[u]:
            # 更新 相连节点的最短距离
            if curDist + w < distance[v]:
                distance[v] = curDist + w
                heapq.heappush(minHeap, (distance[v], v))

    # 如果有节点的最短距离仍为无穷大，说明该节点无法到达
    maxDist = max(distance.values())
    return maxDist if maxDist != float('inf') else -1

#https://leetcode.com/problems/swim-in-rising-water/description/
# On an N x N grid, each square grid[i][j] represents the elevation at that point (i,j).
# Now rain starts to fall. At time t, the depth of the water everywhere is t.
# You can swim from a square to another 4-directionally adjacent square
# 找到从左上角到右下角的最短路径，(因为要水才能游泳）  使得路径上的最大高度尽可能小
"""本质上是一个图的最短路径问题
我们知道最小的成本介于网格中的最小值和最大值之间。我们可以二分这个范围，对于每个给定的高度值，
使用DFS或BFS来检查是否存在一条从左上角到右下角的有效路径，如果存在，尝试降低高度；如果不存在，则增加高度。
下面的算法是Dijkstra
"""
def swimInWater(grid: List[List[int]]) -> int:
    N = len(grid)
    minHeap = [(grid[0][0], 0, 0)]  # (elevation, x, y)
    visited = set((0, 0))
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    maxElev = grid[0][0]

    while minHeap:
        elev, x, y = heapq.heappop(minHeap)
        maxElev = max(maxElev, elev)
        if x == N - 1 and y == N - 1:
            return maxElev
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in visited:
                visited.add((nx, ny))
                heapq.heappush(minHeap, (grid[nx][ny], nx, ny))

    return maxElev

# https://www.lintcode.com/problem/892/
"""the order among letters are unknown to you. You receive a list of non-empty words from the dictionary,
 where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.
 Input：["wrt","wrf","er","ett","rftt"]
Output："wertf"
Explanation：
from "wrt"and"wrf" ,we can get 't'<'f'
from "wrt"and"er" ,we can get 'w'<'e'
from "er"and"ett" ,we can get 'r'<'t'
from "ett"and"rftt" ,we can get 'e'<'r'
So return "wertf"
构建一个有向图，拓扑排序  visiting表示当前正在访问的节点，以检测图中是否存在环；visited表示已经访问完成的节点。
如果在DFS过程中遇到一个正在访问的节点，说明图中存在一个环，这种情况下外星字母的顺序是无法确定的
，我们返回空字符串。否则，一旦完成对所有节点的访问，我们就得到了一个逆序的拓扑排序结果，将其反转后即为正确的字母顺序。
"""
def alienOrder(words: List[str]) -> str:
    # 构建图
    graph = {c: set() for word in words for c in word}
    for i in range(1, len(words)):
        word1, word2 = words[i - 1], words[i]
        for j in range(min(len(word1), len(word2))):
            if word1[j] != word2[j]:
                graph[word1[j]].add(word2[j])
                break

    # 拓扑排序
    result = []
    visited = set()
    visiting = set()

    def dfs(c):
        if c in visiting:
            return False
        if c in visited:
            return True
        visiting.add(c)
        for neighbor in graph[c]:
            if not dfs(neighbor):
                return False
        visiting.remove(c)
        visited.add(c)
        result.append(c)
        return True

    for c in graph:
        if not dfs(c):
            return ""

    return "".join(result[::-1])


#https://leetcode.com/problems/cheapest-flights-within-k-stops/description/
# src, dst, and k, return the cheapest price from src to dst with at most k stops.
"""用广度优先搜索（BFS）或者动态规划（DP）来解决，其中BFS较为直观，而动态规划则更为高效
从src出发，每次飞行可以到达新的城市，直到最多完成k+1次航班（因为从src到dst至少需要一次航班，所以是k+1）。
在这个过程中，我们需要记录并更新到达每个城市的最低成本。
动态规划的思路，我们可以创建一个表来记录到达每个城市所需的最小成本，其中表的行表示经过的航班次数，列表示目的地城市"""
def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    dp = [[float('inf')] * n for _ in range(k + 2)]
    dp[0][src] = 0

    for i in range(1, k + 2):
        dp[i][src] = 0  # 从src出发的成本始终为0
        # dpv = min (dpv, dp u + w
        for u, v, w in flights:
            # 不会溢出，因为 inf + w 仍然是 inf
            dp[i][v] = min(dp[i][v], dp[i - 1][u] + w)

    return dp[k + 1][dst] if dp[k + 1][dst] != float('inf') else -1