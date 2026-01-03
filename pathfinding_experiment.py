import networkx as nx
import random
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
import csv

# Function to create the graph
def generate_graph(num_vertices, density):
    G = nx.DiGraph()
    
    # Create vertices
    for i in range(num_vertices):
        G.add_node(i)
    
    # Determine number of edges
    num_edges = int(density * num_vertices * (num_vertices - 1))
    
    # Create edges with weights
    while G.number_of_edges() < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            weight = random.uniform(1, 10)  # Assign a random weight between 1 and 10
            G.add_edge(u, v, weight=weight)
    
    print(f"Generated graph with {num_vertices} vertices and {G.number_of_edges()} edges.")
    return G, G.number_of_edges()  # Return the graph and number of edges

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# A* Algorithm
def heuristic(node, goal):
    return 0  # Simple placeholder heuristic

def a_star(graph, start, goal):
    open_set = {start}
    came_from = {}
    
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, goal)

    while open_set:
        current = min(open_set, key=lambda node: f_score[node])

        if current == goal:
            return g_score  # or reconstruct the path

        open_set.remove(current)

        for neighbor in graph.neighbors(current):
            weight = graph[current][neighbor]['weight']
            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                open_set.add(neighbor)

    return g_score

# Experiment parameters
num_vertices_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # Example sizes
sparse_density = 0.1  # Adjust for sparse graphs
dense_density = 0.5   # Adjust for dense graphs

# Initialize lists for storing execution times and edges
sparse_dijkstra_times = []
sparse_a_star_times = []
dense_dijkstra_times = []
dense_a_star_times = []
sparse_edges = []
dense_edges = []

# Running the experiment for sparse graphs
for num_vertices in num_vertices_list:
    print(f"Processing Sparse Graph: {num_vertices} vertices")
    # Generate sparse graph
    G_sparse, num_edges_sparse = generate_graph(num_vertices, sparse_density)
    sparse_edges.append(num_edges_sparse)

    # Run Dijkstra's algorithm
    start_time = time.perf_counter()
    dijkstra_results = dijkstra(G_sparse, 0)  # Starting from node 0
    end_time = time.perf_counter()
    sparse_dijkstra_times.append((end_time - start_time) * 1e9)  # Convert to nanoseconds

    # Run A* algorithm (using the last node as goal)
    start_time = time.perf_counter()
    a_star_results = a_star(G_sparse, 0, num_vertices - 1)  # Goal is the last node
    end_time = time.perf_counter()
    sparse_a_star_times.append((end_time - start_time) * 1e9)  # Convert to nanoseconds

# Running the experiment for dense graphs
for num_vertices in num_vertices_list:
    print(f"Processing Dense Graph: {num_vertices} vertices")
    # Generate dense graph
    G_dense, num_edges_dense = generate_graph(num_vertices, dense_density)
    dense_edges.append(num_edges_dense)

    # Run Dijkstra's algorithm
    start_time = time.perf_counter()
    dijkstra_results = dijkstra(G_dense, 0)  # Starting from node 0
    end_time = time.perf_counter()
    dense_dijkstra_times.append((end_time - start_time) * 1e9)  # Convert to nanoseconds

    # Run A* algorithm (using the last node as goal)
    start_time = time.perf_counter()
    a_star_results = a_star(G_dense, 0, num_vertices - 1)  # Goal is the last node
    end_time = time.perf_counter()
    dense_a_star_times.append((end_time - start_time) * 1e9)  # Convert to nanoseconds

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(num_vertices_list, sparse_dijkstra_times, label='Dijkstra (Sparse)', marker='o')
plt.plot(num_vertices_list, sparse_a_star_times, label='A* (Sparse)', marker='o')
plt.plot(num_vertices_list, dense_dijkstra_times, label='Dijkstra (Dense)', marker='s')
plt.plot(num_vertices_list, dense_a_star_times, label='A* (Dense)', marker='s')

plt.xlabel('Number of Vertices')
plt.ylabel('Execution Time (nanoseconds)')
plt.title('Algorithm Performance Comparison for Sparse and Dense Graphs')
plt.legend()
plt.grid()
plt.show()

# Saving results to CSV
with open('algorithm_performance_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number of Vertices', 'Dijkstra Time (nanoseconds)', 'A* Time (nanoseconds)', 'Sparse Edges', 'Dense Edges'])
    for i in range(len(num_vertices_list)):
        writer.writerow([num_vertices_list[i], sparse_dijkstra_times[i], sparse_a_star_times[i], sparse_edges[i], dense_edges[i]])

import csv

# Saving results to CSV
with open('algorithm_performance_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number of Vertices', 
                     'Dijkstra Time (Sparse, nanoseconds)', 
                     'A* Time (Sparse, nanoseconds)', 
                     'Dijkstra Time (Dense, nanoseconds)', 
                     'A* Time (Dense, nanoseconds)'])
    for i in range(len(num_vertices_list)):
        writer.writerow([num_vertices_list[i], 
                         sparse_dijkstra_times[i], 
                         sparse_a_star_times[i], 
                         dense_dijkstra_times[i], 
                         dense_a_star_times[i]])


print("Experiment completed. Results saved to 'algorithm_performance_results.csv'.")
