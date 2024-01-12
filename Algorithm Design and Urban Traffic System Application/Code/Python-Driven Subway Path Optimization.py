import min_heap
import random
import matplotlib.pyplot as plot
import csv
import math
import timeit
from collections import Counter


class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)

    def print_graph(self):
        for i in self.adj.keys():
            print(str(i) + ":",end=" ")
            print(self.adj[i])

def dijkstra_old(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

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
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node

    return dist

def dijkstra_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relax = {}
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        relax[node] = 0
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and relax[current_node] <= k:
                relax[current_node] = relax[current_node] + 1
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist

def bellman_ford(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist

def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    relax = {}
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        relax[node] = 0
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and relax[node] <= k:
                    relax[node] += 1
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total

def create_random_complete_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i,j,random.randint(1,upper))
    return G

def create_random_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            c = random.randint(0,n-1)
            if i != c:
                G.add_edge(i,c,random.randint(1,upper))
    return G

#Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d

def experiment_1a():
    kval = 56   
    runs = 50
    offset = 5
    
    source = 0
    nodes = 50
    upper = 50

    X = [*range(offset,kval+offset)]
    test1 = [0] * kval
    test2 = [0] * kval

    for j in range(runs):
        for i in range(offset,kval+offset):
            G = create_random_complete_graph(nodes,upper)
            test1[i-offset] += total_dist(dijkstra_old(G,source))
            test2[i-offset] += total_dist(dijkstra_approx(G,source,i))

    for x in range(kval):
        test1[x] = test1[x] / runs
        test2[x] = test2[x] / runs

    plot.plot(X, test1, c='blue')
    plot.plot(X, test2, c='red')
    plot.xlabel("Value of k")
    plot.ylabel("Total distance")
    plot.show()

def experiment_1b():
    kval = 46 
    runs = 20
    offset = 5
    
    source = 0
    nodes = 50
    upper = 50

    X = [*range(offset,kval+offset)]
    test1 = [0] * kval
    test2 = [0] * kval

    for j in range(runs):
        for i in range(offset,kval+offset):
            G = create_random_complete_graph(nodes,upper)
            test1[i-offset] += total_dist(bellman_ford(G,source))
            test2[i-offset] += total_dist(bellman_ford_approx(G,source,i))

    for x in range(kval):
        test1[x] = test1[x] / runs
        test2[x] = test2[x] / runs

    plot.plot(X, test1, c='green')
    plot.plot(X, test2, c='orange')
    plot.xlabel("Value of k")
    plot.ylabel("Total distance")
    plot.show()

def experiment_1c():
    kval = 10
    runs = 20
    
    source = 0
    nodes = 50
    upper = 50

    X = [*range(1,nodes)]
    test1 = [0] * (nodes-1)
    test2 = [0] * (nodes-1)

    for j in range(runs):
        for i in range(1,nodes):
            G = create_random_complete_graph(i,upper)
            test1[i-1] += total_dist(dijkstra_old(G, source))
            test2[i-1] += total_dist(dijkstra_approx(G, source, kval))

    for x in range(kval):
        test1[x] = test1[x] / runs
        test2[x] = test2[x] / runs

    plot.plot(X, test1, c='blue')
    plot.plot(X, test2, c='red')
    plot.xlabel("Number of nodes")
    plot.ylabel("Total distance")
    plot.show()

def mysteryPerformanceGraph(maxNodes):
   startT = timeit.default_timer()
   #initialize lists that will store the time data
   dijkstraAvg = []
   bellmanAvg = []
   mysteryAvg = []


   #i is for the number of nodes. We start at 5 nodes, and 2 edges, then add i
   for i  in range(maxNodes):
       dijkstraTimes = []
       bellmanFordTimes = []
       mysteryTimes = []
       reps = 5 #how many reps we take for the average time
       #repeat 'reps' number of times
       for j in range(reps):
           #creat graph. start with 5 nodes and 2 edges
           G = create_random_complete_graph(5 + i, 5 + i)
           #start, then run, then end
           start = timeit.default_timer()
           mystery(G)
           end = timeit.default_timer()
           #then add the time to the list
           mysteryTimes.append(end-start)
           #repeat for each algorithm
           start = timeit.default_timer()
           dijkstra(G,1)
           end = timeit.default_timer()
           dijkstraTimes.append(end-start)
           start = timeit.default_timer()
           bellman_ford(G,1)
           end = timeit.default_timer()
           bellmanFordTimes.append(end-start)
       #after the for loop, calculate all the averages. append them to the avg list
       dijkstraAvg.append(sum(dijkstraTimes)/len(dijkstraTimes))
       bellmanAvg.append(sum(bellmanFordTimes)/len(bellmanFordTimes))
       mysteryAvg.append(sum(mysteryTimes)/len(mysteryTimes))
   endT = timeit.default_timer()

   print('time: ', endT-startT)


   #print out the graph stuff
   overlapping = 0.150
   line1 = plot.plot(dijkstraAvg, c='red', alpha=overlapping, lw=3, label = 'DijkstraAvg')
   line2 = plot.plot(bellmanAvg, c='blue', alpha=overlapping, lw=3, label = 'BellmanAvg')
   line3 = plot.plot(mysteryAvg, c='orange', alpha=overlapping, lw=3, label = 'MysteryAvg')
   plot.title("Performance of different approximations, log/log")
   plot.xlabel("Number of Edges + 5")
   plot.ylabel("Average sum over 5 runs")
   plot.yscale('log')
   plot.xscale('log')
   leg = plot.legend(loc='upper left')
   plot.show()

def dijkstra(G, s, d):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(s, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        if current_node == d:
            break
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node

    return dist[d]

def a_star(G, s, d, h):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
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

stations = {}
connections = {}

with open("./london_stations.csv", 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = int(row.pop('id'))
        stations[key] = row

with open("./london_connections.csv", 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = (int(row.pop('station1')), int(row.pop('station2')))
        connections[key] = row

system = DirectedWeightedGraph()

for station in stations.keys():
    system.add_node(station)

for connection in connections.keys():
    system.add_edge(connection[0],connection[1],int(connections[connection]['time']))
    system.add_edge(connection[1],connection[0],int(connections[connection]['time']))

def hcalc(s,d):
    return math.sqrt(math.pow((float(stations[s]['latitude']) - float(stations[d]['latitude'])),2) + math.pow((float(stations[s]['longitude']) - float(stations[d]['longitude'])),2))

def experiment_2():
    dtime = [0]
    atime = [0]
    t1 = 0
    t2 = 0

    for i in stations.keys():
        for j in stations.keys():
            if i != j:
                heuristic = {}
                for k in stations.keys():
                    heuristic[k] = hcalc(k,j)

                start = timeit.default_timer()
                dijkstra(system, i, j)
                end = timeit.default_timer()
                t1 = end - start

                start = timeit.default_timer()
                a_star(system, i, j, heuristic)
                end = timeit.default_timer()
                t2 = end - start

                print(str(i) + ", " + str(j))
                dtime.append(dtime[-1] + t1)
                atime.append(atime[-1] + t2)

    y1 = plot.plot(dtime, c='blue', alpha=0.7)
    y2 = plot.plot(atime, c='red', alpha=0.7)
    plot.xlabel("Stations")
    plot.ylabel("Cumulative Time")
    plot.show()

experiment_2()