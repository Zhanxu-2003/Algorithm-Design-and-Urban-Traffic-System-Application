import min_heap
import random
import matplotlib.pyplot as plot
import time
import graphInterface
import min_heap
from typing import List

#interface for Shortest path algorithm. For the strategy pattern
class SPAlgorithm:
    def calc_sp(graph: graphInterface.Graph, source: int, dest: int) -> float:
        return 0

class Return1(SPAlgorithm):
    def calc_sp(graph: graphInterface.Graph, source: int, dest: int) -> float:
        return 1.0

#Here we will define the strategies we use
class Dijkstra(SPAlgorithm):
    def calc_sp(graph: graphInterface.Graph, source: int, dest: int):
        pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {} #Distance dictionary
        Q = min_heap.MinHeap([])
        nodes = list(graph.adj.keys())

        #Initialize priority queue/heap and distances
        for node in nodes:
            Q.insert(min_heap.Element(node, float("inf")))
            dist[node] = float("inf")
        Q.decrease_key(source, 0)

    #Meat of the algorithm
        while not Q.is_empty():
            current_element = Q.extract_min()
            current_node = current_element.value
            dist[current_node] = current_element.key
            for neighbour in graph.adj[current_node]:
                if dist[current_node] + graph.w(current_node, neighbour) < dist[neighbour]:
                    Q.decrease_key(neighbour, dist[current_node] + graph.w(current_node, neighbour))
                    dist[neighbour] = dist[current_node] + graph.w(current_node, neighbour)
                    pred[neighbour] = current_node
                if neighbour == dest:
                    break
        return dist[dest]

class Bellman_Ford(SPAlgorithm):
    def calc_sp(graph: graphInterface.Graph, source: int, dest: int) -> float:
        pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {} #Distance dictionary
        nodes = list(graph.adj.keys())

        #Initialize distances
        for node in nodes:
            dist[node] = float("inf")
        dist[source] = 0

        #Meat of the algorithm
        for _ in range(len(graph.adj)): #####Separation Violation##### in order to keep more to the specs, we expose the graph's adj matrix
            for node in nodes:
                for neighbour in graph.adj[node]:
                    if dist[neighbour] > dist[node] + graph.w(node, neighbour):
                        dist[neighbour] = dist[node] + graph.w(node, neighbour)
                        pred[neighbour] = node
                    if neighbour == dest:
                        break
        return dist[dest]
        
        return super().calc_sp(source, dest)
    
class A_Star_Adapter(SPAlgorithm):
   def __init__(self):
       super().__init__()


   def calc_sp(self, graph: graphInterface.HeuristicGraph, source, dest):
       # Convert the given graph to a compatible format for the A_Star class
       
       # You will need to implement a function `convert_graph_to_compatible_format` to do this conversion
       #converted_graph = self.convert_graph_to_compatible_format(graph, dest)
      
       # Call the A_Star algorithm
       a_star_instance = A_Star(graph, self.heuristic)
       result = a_star_instance.run_a_star(source, dest)
       shortest_path_distance = result[1]
      
       return shortest_path_distance


   def convert_graph_to_compatible_format(self, graph):
       # Implement this function to convert the given graph to a format compatible with the A_Star class
       pass


class A_Star:
   def __init__(self, graph, heuristic):
       self.graph = graph
       self.heuristic = heuristic

   def run_a_star(self, source, dest):
       # The provided A* code is placed here
       G = self.graph
       s = source
       d = dest
       h = self.heuristic
       pred = {}
       dist = {}
       Q = min_heap.MinHeap([])
       nodes = list(G.adj.keys())


       #Initialize priority queue/heap and distances
       for node in nodes:
           Q.insert(min_heap.Element(node, float("inf")))
           dist[node] = float("inf")
  
       dist[s] = 0
       Q.decrease_key(s, 0)


       #Meat of the algorithm
       while not Q.is_empty():
           current_element = Q.extract_min()
           current_node = current_element.value


           if current_node == d:
               break
           for neighbour in G.adj[current_node]:
               if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                   dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                   Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour) + h[neighbour])
                   pred[neighbour] = current_node
       return (pred, dist[d])


