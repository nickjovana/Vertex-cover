import timeit
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# function removes node from the graph
def remove_from_graph(graph, node):
	for k, v in graph.items():
		if (k == node):
			for edge in v:
				v.remove(edge)
				graph = remove_from_graph(graph, node)
		if node in v:
			v.remove(node)
			graph = remove_from_graph(graph, node)

	return graph

# function checks values of degrees for all nodes in graph
# and returns True if all are equal 0, and False othervise
def check_degrees(degrees):
	length = len(degrees)
	i = 0
	for value in degrees:
		if value[1] == 0:
			i += 1

	if (i == length):
		return True
	else:
		return False

# function implements greedy algorithm and returns vertex cover
def greedy(graph, g):
	degrees = sorted(g.degree(), key = lambda x: x[1], reverse=True)
	cover = []
	for k, v in graph.items():
		m = degrees[0]
		cover.append(m[0])
		graph = remove_from_graph(graph, m[0])
		g = nx.Graph(graph)
		degrees = sorted(g.degree(), key = lambda x: x[1], reverse=True)
		if (check_degrees(degrees) == True):
			break

	return len(cover)

def main():
	graf1 = nx.connected_watts_strogatz_graph(5, 4, 0.5, seed = 1)
	graf2 = nx.connected_watts_strogatz_graph(50, 20, 0.5, seed = 1)
	graf3 = nx.connected_watts_strogatz_graph(1000, 50, 0.5, seed = 1)
	graph = nx.to_dict_of_lists(graf3)
	g = nx.Graph(graph)
	start = timeit.default_timer()
	print("Vertex cover: ", greedy(graph, g))
	stop = timeit.default_timer()
	print('Time: ', stop - start)

if __name__ == "__main__":
	main()
