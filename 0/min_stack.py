class MinStack:
    # 核心思路是用两个栈，一个栈用来存储数据，另一个栈用来存储最小值
    # 每次push的时候，如果当前值比最小值栈的栈顶元素小，就push到最小值栈中
    # pop的时候，如果pop出来的值等于最小值栈的栈顶元素，就pop最小值栈
    # 这样最小值栈的栈顶元素始终是当前最小值
    def __init__(self):
        self.st=[]
        self.min=[float('inf')]

    def push(self, val: int) -> None:
        self.st.append(val)
        self.min.append(min(self.min[-1],val))

    def pop(self) -> None:
        self.min.pop()
        self.st.pop()

    def top(self) -> int:
        return self.st[-1]

    def getMin(self) -> int:
        return self.min[-1]