import matplotlib.pyplot as plot
import graphInterface
import SPAInterface
import min_heap
import random

#main class
class ShortestPathFinder:
    def __init__(self):
        self.algorithm = None
        self.graph = None

    def calc_short_path(self, source: int, dest: int) -> float:
        return self.algorithm.calc_sp( self.graph, source, dest)
    
    def set_graph(self, graph: graphInterface.Graph):
        self.graph = graph
    
    def set_algorithm(self, algorithm):#: SPAInterface.SPAlgorithm):
        self.algorithm = algorithm

    
def main():
    s = ShortestPathFinder
    s.set_algorithm(s,SPAInterface.Dijkstra)
    s.set_graph(s,graphInterface.Weightedgraph())
    

    upper = 6
    nodes = 8

    
    #create graph
    print("start")
    s.graph.print()

    for i in range(nodes):
        s.graph.add_node(i)
        #s.graph.print()
    for i in range(nodes):
        for j in range(nodes):
            r = random.randint(1,2)#this is optional
            if i != j and r == 1:#it just makes the graph less complete
                s.graph.add_edge(i,j,random.random()*upper)
 #               s.graph.print()


    #s.set_algorithm(s,SPAInterface.A_Star_Adapter(s.graph.heuristic))
    

    print('shortest path distance: ', s.calc_short_path(s,1,7))
    print('graph: ')
    s.graph.print()
main()