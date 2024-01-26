from collections import defaultdict
from typing import List
from trie import TrieNode, Trie


# word may contain dots '.' where dots can be matched with any letter.
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.children[char]
        node.isEndOfWord = True

    def search(self, word: str) -> bool:
        def search_in_node(word, node):
            for i, char in enumerate(word):
                if char not in node.children:
                    # If the current character is '.', check all possible nodes at this level
                    if char == '.':
                        for x in node.children.values():
                            if search_in_node(word[i+1:], x):
                                return True
                    # If no match is found, return False
                    return False
                else:
                    # Move to the next node in the Trie
                    node = node.children[char]
            return node.isEndOfWord
        return search_in_node(word, self.root)

def dfs(board, i, j, node, word, result):
    if node.isEndOfWord:
        result.add(word)
        node.isEndOfWord = False  # To avoid re-adding the same word

    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] not in node.children:
        return

    char = board[i][j]
    board[i][j] = "#"  # Mark as visited
    for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        ni, nj = i + x, j + y
        dfs(board, ni, nj, node.children[char], word + char, result)
    board[i][j] = char  # Reset after visiting

#https://leetcode.com/problems/word-search-ii/description/ 用trie不用重复搜索
def findWords(board: List[List[str]], words: List[str]) -> List[str]:
    trie = Trie()
    for word in words:
        trie.addWord(word)
    result = set()
    for i in range(len(board)):
        for j in range(len(board[0])):
            dfs(board, i, j, trie.root, "", result)
    return list(result)

def exist(board: List[List[str]], word: str) -> bool:
    def dfs(i, j, k):
        if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
            return False
        if k == len(word) - 1:
            return True
        tmp, board[i][j] = board[i][j], '/'
        res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
        board[i][j] = tmp
        return res

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(i, j, 0):
                return True
    return False

# Usage example
if __name__ == "__main__":
    wordDictionary = WordDictionary()
    wordDictionary.addWord("bad")
    wordDictionary.addWord("dad")
    wordDictionary.addWord("mad")
    print(wordDictionary.search("pad"))  # Returns False
    print(wordDictionary.search("bad"))  # Returns True
    print(wordDictionary.search(".ad"))  # Returns True
    print(wordDictionary.search("b.."))  # Returns True

    board = [
        ['o','a','a','n'],
        ['e','t','a','e'],
        ['i','h','k','r'],
        ['i','f','l','v']
    ]
    words = ["oath", "pea", "eat", "rain"]
    print(findWords(board, words))