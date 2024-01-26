#https://leetcode.com/problems/design-twitter/
import heapq
import itertools
from collections import defaultdict, deque
from typing import List

class Tweet():
    def __init__(self, idx, time):
        self.idx = idx
        self.time = time

    def __lt__(self, other):
        return self.time < other.time

    def getId(self) -> int:
        return self.idx


class Twitter:
    def __init__(self):
        self.__feedLimit = 10
        # unique id for tweet according time, here also can be used timestamp or more advanced UUID https://docs.python.org/3/library/uuid.html
        self.uuid = itertools.count()                                     # it's just a number 1,2,3....
        self.subscribers = defaultdict(set)                               # userId => set(subscribers1, subscriber2, ...)
        self.subscriptions = defaultdict(set)                             # userId => set(subscription1, subscription2, ...)
        self.feed = defaultdict(lambda: deque(maxlen = self.__feedLimit)) # userId => deque(feed)
        self.posts = defaultdict(deque)                                   # userId => deque(user posts, newest from the left), or DoubleLinkelList if needed deletion support

    def postTweet(self, userId: int, tweetId: int) -> None:
        tweet = Tweet(tweetId, next(self.uuid))
        self.posts[userId].appendleft(tweet)
        # 合并 遍历 自己和自己的订阅者，将新的tweet加入到他们的feed中
        for idx in itertools.chain([userId], self.subscribers[userId]):   # add tweet to all subscribers feeds
            self.__addToFeed(idx, tweet)

    def getNewsFeed(self, userId: int) -> List[int]:
        return list(map(Tweet.getId, self.feed[userId]))

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.subscribers[followeeId]: return
        self.subscribers[followeeId].add(followerId)
        self.subscriptions[followerId].add(followeeId)
        self.__addUserToFeed(followerId, followeeId) # here you can choose this special method or __updateUserFeed()

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId not in self.subscribers[followeeId]: return
        self.subscribers[followeeId].discard(followerId)
        self.subscriptions[followerId].discard(followeeId)
        self.__updateUserFeed(followerId)

    def __addToFeed(self, userId: int, tweet: Tweet, toStart = True) -> None:
        if toStart:
            self.feed[userId].appendleft(tweet)
        else:
            self.feed[userId].append(tweet)

    def __clearFeed(self, userId: int) -> None:
        self.feed[userId].clear()

    # it's max 20 tweets, so instead of calling __updateUserFeed() can be used simple sorting of current feed and new user feed
    def __addUserToFeed(self, userId, addUserId):
        feed = sorted(list(self.feed[userId]) + list(self.posts[addUserId]), reverse = True)[:self.__feedLimit]
        self.__clearFeed(userId)
        for tweet in feed:
            self.__addToFeed(userId, tweet, False)

        # An alternative approach with eliminating old tweets over limit using automatic feed slicing (deque max len)
        #feed = sorted(list(self.feed[userId]) + list(self.posts[addUserId]))
        #self.__clearFeed(userId)
        #list(map(lambda tweet: self.__addToFeed(userId, tweet), feed))

    def __updateUserFeed(self, userId) -> None:
        self.__clearFeed(userId)
        # 合并当前用户（userId）和他所订阅的所有用户的推文，并按时间顺序（最新的在前）排序，最后获取前 self.__feedLimit 条推文
        # * 用于解包列表，使其成为 heapq.merge 的多个参数
        tweets = heapq.merge(*(self.posts[idx] for idx in self.subscriptions[userId] | {userId}), reverse = True)
        # 生成迭代器，获取tweets 前 self.__feedLimit 个元素
        limitTweets = itertools.islice(tweets, self.__feedLimit)

        for tweet in limitTweets:
            self.__addToFeed(userId, tweet, False)
