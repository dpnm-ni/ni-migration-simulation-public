import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from core.machine import Machine


# TODO: implement as singleton
class MECNetwork:
    def __init__(self):
        # self.topology_graph, self.topology_json = self.create_topology("graph/Abilene1.gml")
        self.topology_graph, self.topology_json = self.create_topology("graph/Kreonet1.gml")
        self.machines = []
        # self.core_machines = []
        # self.edge_machines = []
        self.services = []

    def create_topology(self, graph_file):
        G = self.assign_link_weight_by_distance(nx.read_gml(path=graph_file))
        print(f"nodes: {G.nodes(data=True)}")
        # print(f"edges: {G.edges(data=True)}")
        # print(dict(nx.all_pairs_dijkstra(G)))
        print(dict(nx.all_pairs_dijkstra_path(G)))
        print(dict(nx.all_pairs_dijkstra_path_length(G)))

        topology_json = nx.node_link_data(G)
        del topology_json["directed"]
        del topology_json["multigraph"]
        del topology_json["graph"]
        # print(topology_json)

        return G, topology_json

    # https://stackoverflow.com/questions/64945985/trying-to-find-the-distance-euclidean-between-two-nodes-using-networkx/64946245#64946245
    def assign_link_weight_by_distance(self, G):
        for link in G.edges:
            # print(link)
            source = link[0]
            dest = link[1]
            x1, y1 = G.nodes[source]['Longitude'], G.nodes[source]['Latitude']
            x2, y2 = G.nodes[dest]['Longitude'], G.nodes[dest]['Latitude']
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # print("euclidean distance between {} and {}: {}".format(source, dest, distance))
            G[source][dest]['weight'] = np.rint(distance)

        return G

    def draw_topology(self):
        # Networkx drawing graph with position info
        # https://stackoverflow.com/questions/62999488/plot-networkx-graph-with-coordinates/63009048#63009048
        pos = {}
        NDV = self.topology_graph.nodes(data=True)
        for n, dd in NDV:
            pos[n] = (dd.get("Longitude"), dd.get("Latitude"))
        # print(pos)

        nx.draw(self.topology_graph, pos=pos, with_labels=True)
        # https://stackoverflow.com/questions/57421372/display-edge-weights-on-networkx-graph
        labels = {e: self.topology_graph.edges[e]['weight'] for e in self.topology_graph.edges}
        nx.draw_networkx_edge_labels(self.topology_graph, pos=pos, edge_labels=labels)
        plt.show()

    def get_path_cost(self, source_id, dest_id):
        source = "server{}".format(source_id)
        dest = "server{}".format(dest_id)
        return nx.dijkstra_path_length(self.topology_graph, source, dest)

    def get_least_cost_dest(self, source_id, machine_ids):
        source_id = "server{}".format(source_id)
        machine_ids = ["server{}".format(machine_id) for machine_id in machine_ids]
        dict_dest_cost = dict(nx.all_pairs_dijkstra_path_length(self.topology_graph))[source_id]
        # https://stackoverflow.com/questions/32727294/remove-keys-from-object-not-in-a-list-in-python
        dict_dest_cost = {k: dict_dest_cost[k] for k in machine_ids if k in dict_dest_cost}
        # print(dict_dest_cost)
        # https://gomguard.tistory.com/137
        least_cost_dest = min(dict_dest_cost.keys(), key=(lambda k: dict_dest_cost[k]))
        # print(least_cost_dest)

        return int(least_cost_dest[len("server"):])

    def add_machines(self, machine_profiles):
        for machine_profile in machine_profiles:
            machine = Machine(machine_profile)
            self.machines.append(machine)

    # register a service request
    def add_service(self, service):
        self.services.append(service)

    def get_waiting_services(self):
        ls = []
        for service in self.services:
            if service.is_started() is False:
                ls.append(service)
        return ls

    def get_unfinished_services(self):
        ls = []
        for service in self.services:
            if service.is_started() is True and service.is_finished() is False:
                ls.append(service)
        return ls


def test():
    mec_net = MECNetwork()
    mec_net.draw_topology()


if __name__ == '__main__':
    os.chdir("..")
    test()
