from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isEndOfWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.children[char]
        node.isEndOfWord = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            node = node.children.get(char)
            if node is None:
                return False
        return node.isEndOfWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            node = node.children.get(char)
            if node is None:
                return False
        return True

# Unit tests
if __name__ == "__main__":
    trie = Trie()
    words = ["hello", "world", "hell", "word"]
    for word in words:
        trie.insert(word)

    assert trie.search("hello") == True, "Error in search method."
    assert trie.search("world") == True, "Error in search method."
    assert trie.search("hell") == True, "Error in search method."
    assert trie.search("helloo") == False, "Error in search method."

    assert trie.startsWith("wo") == True, "Error in startsWith method."
    assert trie.startsWith("h") == True, "Error in startsWith method."
    assert trie.startsWith("he") == True, "Error in startsWith method."
    assert trie.startsWith("xyz") == False, "Error in startsWith method."

    print("All tests passed!")
