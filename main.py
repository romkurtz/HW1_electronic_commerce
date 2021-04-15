from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy

PART_B_C = "PartB-C.csv"


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    for node in graph:
        graph.nodes[node]['concern'] = 0
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

        # step 3 : every v in total_suspected updates her concern
        for node in total_suspected:
            graph.nodes[node]['concern'] = calc_concern_LTM(set(graph.neighbors(node)), total_infected)

        total_infected = total_infected.union(new_infected)

    return total_infected


def calc_concern_LTM(neighbors: set, patients: set) -> float:
    return len(neighbors.intersection(patients)) / len(neighbors)


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_deceased = set()
    deceased_t_minus_1 = set()
    infected_t_minus_1 = set()
    for node in graph:
        graph.nodes[node]['concern'] = 0
    total_infected = set(patients_0)
    for node in total_infected:
        if np.random.random() < LETHALITY:
            deceased_t_minus_1.add(node)
            total_deceased.add(node)
            graph.remove_node(node)

    total_suspected = set(graph.nodes).difference(total_infected)

    for t in range(iterations):
        new_infected = set()
        new_deceased = set()
        for node in total_suspected:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor in infected_t_minus_1:
                    prob = min(1, CONTAGION*graph.edges[node][neighbor]['weight']*(1-graph.nodes[node]['concern']))
                    if np.random.random() < prob:
                        new_infected.add(node)

        for node in new_infected:
            if np.random.random() < LETHALITY:
                new_deceased.add(node)
                total_deceased.add(node)
                graph.remove_node(node)

        total_infected = (total_infected.union(new_infected)).difference(total_deceased)
        total_suspected = total_suspected.difference((total_infected.union(total_deceased)))

        for node in total_suspected:
            graph.nodes[node]['concern'] = calc_concern_ICM(set(graph.neighbors(node)), infected_t_minus_1, deceased_t_minus_1)

        infected_t_minus_1 = new_infected
        deceased_t_minus_1 = new_deceased

    return total_infected, total_deceased


def calc_concern_ICM(neighbors: set, patients: set, deceased: set) -> float:
    numerator = len(neighbors.intersection(patients)) + 3*len(neighbors.intersection(deceased))
    denominator = len(neighbors)
    return min(numerator / denominator, 1)



def plot_degree_histogram(histogram: Dict):
    width = 1
    plt.bar(histogram.keys(), histogram.values(), width, color='g')
    plt.title('Degree histogram')
    plt.xlabel('degree')
    plt.ylabel('# nodes')
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
        total_deaths = 0
        total_infected = 0
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            infected, deceased = ICM(graph, patients_0, t)
            total_deaths += len(deceased)
            total_infected += len(infected)

        mean_deaths[l] = total_deaths / 30
        mean_infected[l] = total_infected / 30

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    plt.plot(mean_infected.keys(), mean_infected.values(), label='infected')
    plt.plot(mean_deaths.keys(), mean_deaths.values(), label='death')
    plt.legend()
    plt.title('Lethality effect on death and infected')
    plt.xlabel('Lethality')
    plt.ylabel('mean')
    plt.show()


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


def test_plot_2_dict():
    d1 = {0: 20, 1: 30, 2: 40, 3: 50}
    d2 = {0: 12, 1: 25, 2: 45, 3: 60}
    plot_lethality_effect(d1, d2)


"Global Hyper-parameters"
CONTAGION = 1
LETHALITY = .15

if __name__ == "__main__":
    test_plot_2_dict()
    filename = "PartB-C.csv"
    G = build_graph(filename=filename)
    plot_degree_histogram(calc_degree_histogram(G))
    """df = pd.read_csv('patients0.csv')
    df_lst = [val[0] for val in df.values.tolist()]
    bb = len(LTM(G, df_lst[:50], 6))
    print(bb)
    CC = clustering_coefficient(G)
    print(CC)
    hist = calc_degree_histogram(G)
    plot_degree_histogram(hist)"""



