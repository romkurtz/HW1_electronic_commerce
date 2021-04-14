from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy

PART_B_C = "PartB-C.csv"


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    for node in graph:
        graph.nodes[node]['concern'] = calc_concern(set(graph.neighbors(node)), set(patients_0))
    total_infected = set(patients_0)
    total_suspected = set(graph.nodes).difference(total_infected)

    for t in range(iterations):
        # step 1 : every node in total_suspected watches all of her infected neighbors and perform contagion if needed.
        new_infected = set()
        for node in total_suspected:
            infected_neighbors = set(graph.neighbors(node)).intersection(total_infected)
            weight_edges_sum = 0
            for neighbor in infected_neighbors:
                weight_edges_sum += graph.edges[node, neighbor]['weight']

            if CONTAGION * weight_edges_sum >= 1 + graph.nodes[node]['concern']:
                new_infected.add(node)

        # step 2 : total_suspected = total_suspected \ new_infected and update total_infected
        total_suspected = total_suspected.difference(new_infected)
        total_infected = total_infected.union(new_infected)

        # step 3 : every v in total_suspected updates her concern
        for node in total_suspected:
            graph.nodes[node]['concern'] = calc_concern(set(graph.neighbors(node)), total_infected)

    return total_infected


def calc_concern(neighbors: set, patients: set) -> float:
    return len(neighbors.intersection(patients)) / len(neighbors)


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_infected = set(patients_0)
    total_deceased = set()
    # TODO implement your code here
    return total_infected, total_deceased


def plot_degree_histogram(histogram: Dict):
    width = 1
    plt.bar(histogram.keys(), histogram.values(), width, color='g')
    plt.show()


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    #return graph.degree
    hist = graph.degree
    degrees_lst = list()
    degrees_dict = dict()
    for item in hist:
        degrees_lst.append(item[1])
    for i in range(max(degrees_lst) + 1):
        degrees_dict[i] = 0
    for item in hist:
        degrees_dict[item[1]] += 1

    return degrees_dict



    histogram = {}
    # TODO implement your code here
    return histogram


def build_graph(filename: str) -> networkx.Graph:

    if filename == PART_B_C:
        df = pd.read_csv(filename)
        G = networkx.Graph()
        for index, row in df.iterrows():
            G.add_edge(int(row['from']), int(row['to']), weight=row['w'])

    else:
        G = networkx.read_edgelist(filename, delimiter=",")
        G.remove_node("from")
        G.remove_node("to")

    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    numerator = sum(networkx.triangles(graph).values())
    denominator = float(calc_triplets(graph))
    return numerator / denominator


def calc_triplets(graph: networkx.Graph) -> float:
    sum_triplets = 0
    for node in graph.nodes:
        num_of_neigbhors = len(graph.adj[node])
        sum_triplets += calc_choose(num_of_neigbhors, 2)

    return sum_triplets


def calc_choose(up: int, down: int) -> float:
    return np.math.factorial(up) / (np.math.factorial(down) * np.math.factorial(up - down)) if up > 1 else 0



def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            # TODO implement your code here

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    # TODO implement your code here
    ...


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    people_to_vaccinate = []
    # TODO implement your code here
    return people_to_vaccinate


def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    """
    The following heuristic for Part C is simply taking the top 50 friendly people;
     that is, it returns the top 50 nodes in the graph with the highest degree.
    """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = 1
LETHALITY = .15

if __name__ == "__main__":
    filename = "PartB-C.csv"
    G = build_graph(filename=filename)
    df = pd.read_csv('patients0.csv')
    df_lst = [val[0] for val in df.values.tolist()]
    bb = len(LTM(G, df_lst[:48], 6))
    print(bb)
    #CC = clustering_coefficient(G)
    #print(CC)
    #hist = calc_degree_histogram(G)
    #plot_degree_histogram(hist)



