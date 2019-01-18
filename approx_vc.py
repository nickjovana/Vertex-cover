import timeit
import networkx as nx
import matplotlib.pyplot as plt

# function removes node and value from graph and returns
# new graph
def delete_from_graph(graph, node, value):
	for k, v in graph.items():
		if (k == node or k == value):
			for edge in v:
				v.remove(edge)
				graph = delete_from_graph(graph, node, value)
		if node in v:
			v.remove(node)
			graph = delete_from_graph(graph, node, value)
		if value in v:
			v.remove(value)
			graph = delete_from_graph(graph, node, value)

	return graph

# function implements approximate algorithm and returns
# vertex cover
def approx_vertex_cover(graph):
	marked = []

	for k, v in graph.items():
		for node in v:
			marked.append((k, node))
			graph = delete_from_graph(graph, k, node)

	if (len(marked) == 1):
		return 2

	return len(marked)

def main():
	graf1 = nx.connected_watts_strogatz_graph(5, 4, 0.5, seed = 1)
	graf2 = nx.connected_watts_strogatz_graph(50, 20, 0.5, seed = 1)
	graf3 = nx.connected_watts_strogatz_graph(1000, 50, 0.5, seed = 1)
	graph = nx.to_dict_of_lists(graf3)
	g = nx.Graph(graph)
	start = timeit.default_timer()
	print("Vertex cover: ", approx_vertex_cover(graph))
	stop = timeit.default_timer()
	print('Time: ', stop - start)
	nx.draw(g, with_labels = True)
	plt.show()

if __name__ == "__main__":
	main()
