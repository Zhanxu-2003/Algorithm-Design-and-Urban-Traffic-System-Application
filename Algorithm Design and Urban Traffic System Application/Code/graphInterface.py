from typing import List
#interface for graph
class Graph:
    def __init__(self):
        self.adj = {}

    def get_adj_nodes(self, node: int) -> List[int]:
        return self.adj[node]

    def add_node(self, node: int):
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, start: int, end:int, w:float): #####Figure out what to do with this 'w' variable for unweighted graphs. Do method overload or soemthing 
        if start not in self.adj[end]:
            self.adj[start].append(end)
            self.adj[end].append(start)

    def get_num_nodes(self) -> int:
        return len(self.adj)
    
    #on an unweighted graph, this should return 1.0 for all weights
    def w(self, node: int) -> float:
        return 1.0
    def w(self, node: int, node2: int) -> float:
        return 1.0
    
    def print(self):
        #print graph
        n = self.get_num_nodes()
        for i in range(n):
            print(self.adj[i])

#now for the weighted graph
class Weightedgraph(Graph):
    
    #will need a new initialization method to handle weights
    def __init__(self):
        super().__init__()# Call the parent class's __init__ method
        self.weights = {}#an unweighted graph will still have weights, they will all just be 1.0

    #this needs to actually add the weight, instead of just setting it to 1.0
    def add_edge(self, start: int, end:int, w:float):
        if end not in self.adj[start]:
            self.adj[start].append(end)
            self.adj[end].append(start)
        self.weights[(start, end)] = w
        self.weights[(end, start)] = w

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]
        
    def are_connected(self, node1, node2):
        if node2 in self.adj[node1]:
            return True
        return False

#now Heuristic graph
class HeuristicGraph(Weightedgraph):
    def __init__(self):
        super().__init__()
        self.heuristic = {}

    def add_node(self, node: int, h: float):
        super().add_node(node)
        self.heuristic[node] = h

    def add_edge(self, start: int, end: int, w: float, h: float):
        super().add_edge(start, end, w)
        self.heuristic[start] = h

    def get_heuristic(self, node: int) -> float:
        return self.heuristic.get(node, 0.0)

