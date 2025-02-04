import math
class MinHeap:
    length = 0
    data = []
    
    # 初始化这个heap通过list
    def __init__(self, L):
        self.data = L
        self.length = len(L)
        self.map = {}
        for i in range(len(L)):
            self.map[L[i].value] = i
        self.build_heap()
    
    # 承接步骤
    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink(i)
    
    # 通过这一步真正的构造出一个完整的heap，跟sink_down一样（从上往下）
    def sink(self, i):
        smallest_known = i
        if self.left(i) < self.length and self.data[self.left(i)].key < self.data[i].key:
            smallest_known = self.left(i)
        if self.right(i) < self.length and self.data[self.right(i)].key < self.data[smallest_known].key:
            smallest_known = self.right(i)
        if smallest_known != i:
            self.data[i], self.data[smallest_known] = self.data[smallest_known], self.data[i]
            self.map[self.data[i].value] = i
            self.map[self.data[smallest_known].value] = smallest_known
            self.sink(smallest_known)
    
    # 添加一个数值
    def insert(self, element):
        if len(self.data) == self.length:
            self.data.append(element)
        else:
            self.data[self.length] = element
        self.map[element.value] = self.length
        self.length += 1
        self.swim(self.length - 1)
   
    # 添加一个list的数值
    def insert_elements(self, L):
        for element in L:
            self.insert(element)
    
    # 通过swap的形式把添加的数字放到他应该出现的位置（从下往上）
    def swim(self, i):
        while i > 0 and self.data[i].key < self.data[self.parent(i)].key:
            self.data[i], self.data[self.parent(i)] = self.data[self.parent(i)], self.data[i]
            self.map[self.data[i].value] = i
            self.map[self.data[self.parent(i)].value] = self.parent(i)
            i = self.parent(i)
    
    # 返回最小的数值
    def get_min(self):
        if len(self.data) > 0:
            return self.data[0]

    # 找出最小值，返回并可把最小值给删掉，为heapsort做铺垫
    def extract_min(self):
        self.data[0], self.data[self.length - 1] = self.data[self.length - 1], self.data[0]
        self.map[self.data[self.length - 1].value] = self.length - 1
        self.map[self.data[0].value] = 0
        min_element = self.data[self.length - 1]
        self.length -= 1
        self.map.pop(min_element.value)
        self.sink(0)
        return min_element
    
    # 如果new key的值大于等于value.key，返回none。其他的就把原来的数变成新的数，在swim up
    def decrease_key(self, value, new_key):
        if new_key >= self.data[self.map[value]].key:
            return
        index = self.map[value]
        self.data[index].key = new_key
        self.swim(index)
    
    # 通过输入value，得到整个element
    def get_element_from_value(self, value):
        return self.data[self.map[value]]
    # 检查是否empty
    def is_empty(self):
        return self.length == 0

    def left(self, i):
        return 2 * (i + 1) - 1

    def right(self, i):
        return 2 * (i + 1)

    def parent(self, i):
        return (i + 1) // 2 - 1

    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.data[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s



class Element:

    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __str__(self):
        return "(" + str(self.value) + "," + str(self.key) + ")"

nodes1 = [Element("A", 5), Element("B", 1), Element("C", 10), Element("D", 2), Element("E", -3)]
nodes2 = [Element(1, 1), Element(2, 1), Element(3, 10), Element(4, 2), Element(5, -3)]


heap = MinHeap(nodes2)
heap2 = MinHeap(nodes1)
print(heap)
print(heap.map)
heap.decrease_key(4,-1)
print(heap)
print(heap.map)

print(heap2.get_element_from_value("A"))
a = heap.extract_min()
b = a.key
c = a.value
print(b)
print(c)
print(heap.extract_min())

print("121333")

















