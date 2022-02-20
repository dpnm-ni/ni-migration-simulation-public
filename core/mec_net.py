import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from config import cfg
from core.edge_dc import LeafSpineEdgeDC, SimpleEdgeDC, ThreeTierEdgeDC
from core.machine import Machine


# TODO: implement as singleton
class MECNetwork:
    def __init__(self):
        self.machines = []
        self.services = []
        self.max_path_cost = 0
        # for debug purpose
        self.interrupted_services = []

        self.topo = self.create_topology("graph/Abilene1.gml")
        # self.topo = self.create_topology("graph/Kreonet1.gml")

        self.edgeDCs = []
        # dictionary for computing path costs between edge DCs
        self.edgeDC_topos = {}

    def create_topology(self, graph_file):
        G = self.assign_link_weight_by_distance(nx.read_gml(path=graph_file))
        print(f"nodes: {G.nodes(data=True)}")
        print(f"edges: {G.edges(data=True)}")
        # print(dict(nx.all_pairs_dijkstra(G)))
        # print(dict(nx.all_pairs_dijkstra_path(G)))
        # print(dict(nx.all_pairs_dijkstra_path_length(G)))

        # set maximum path cost which is used in reward function in Algorithm
        self.max_path_cost = self.get_max_path_cost(G)

        # topology_json = nx.node_link_data(G)
        # del topology_json["directed"]
        # del topology_json["multigraph"]
        # del topology_json["graph"]
        # print(topology_json)

        return G

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

    def create_edgeDCs(self):
        for _ in range(cfg["mec_net"]["num_simple_edges"]):
            edgeDC = SimpleEdgeDC()
            edgeDC.create_machines(self)
            self.edgeDCs.append(edgeDC)

        for _ in range(cfg["mec_net"]["num_leaf_spine_edges"]):
            edgeDC = LeafSpineEdgeDC()
            edgeDC.create_machines(self)
            self.edgeDCs.append(edgeDC)

        for _ in range(cfg["mec_net"]["num_three_tier_edges"]):
            edgeDC = ThreeTierEdgeDC()
            edgeDC.create_machines(self)
            self.edgeDCs.append(edgeDC)

        # for i in range(len(self.edgeDCs)):
        #     self.edgeDC_topos[self.edgeDCs[i].id] = self.edgeDCs[i].topo

    # replace initial nodes created by graph file with corresponding edge DCs we just create
    def apply_edgeDCs(self):
        dict_node_to_replace = dict(zip(self.topo.nodes, self.edgeDCs))
        for node in dict_node_to_replace.keys():
            edge_dc = dict_node_to_replace[node].topo
            longitude = self.topo.nodes[node]["Longitude"]
            latitude = self.topo.nodes[node]["Latitude"]

            # https://networkx.org/documentation/stable/tutorial.html#nodes
            # contains the nodes of edge1 as nodes of mec_net
            # mec_net.topo.add_nodes_from(edge_dc, Longitude=longitude, Latitude=latitude)
            # vs. contains edge1 as a node of mec_net
            self.topo.add_node(edge_dc, Longitude=longitude, Latitude=latitude)

            # replace related links (does not mean edge DCs)
            edges = self.topo.edges(node)
            for edge in edges:
                weight = self.topo.edges[(edge[0], edge[1])]["weight"]
                self.topo.add_edge(edge_dc, edge[1], weight=weight)
            self.topo.remove_node(node)

        # self.draw_topology()

    def draw_topology(self):
        # Networkx drawing graph with position info
        # https://stackoverflow.com/questions/62999488/plot-networkx-graph-with-coordinates/63009048#63009048
        pos = {}
        NDV = self.topo.nodes(data=True)
        for n, dd in NDV:
            pos[n] = (dd.get("Longitude"), dd.get("Latitude"))
        # print(pos)

        nx.draw(self.topo, pos=pos, with_labels=True)
        # https://stackoverflow.com/questions/57421372/display-edge-weights-on-networkx-graph
        labels = {e: self.topo.edges[e]['weight'] for e in self.topo.edges}
        nx.draw_networkx_edge_labels(self.topo, pos=pos, edge_labels=labels)
        plt.show()

    def get_max_path_cost(self, G):
        dict_dest_cost = dict(nx.all_pairs_dijkstra_path_length(G))
        max_path_cost = 0
        for key in dict_dest_cost.keys():
            path_cost = max(dict_dest_cost[key].values())
            if path_cost > max_path_cost:
                max_path_cost = path_cost
        # print(max_path_cost)
        return max_path_cost

    # !source_id: service.user_loc (so edge DC id)
    # !dest_id: destination machine id
    def get_path_cost(self, source_id, dest_id):
        source = self.edgeDCs[source_id]
        dest_edgeDC_id = self.machines[dest_id].machine_profile.edgeDC_id
        dest = self.edgeDCs[dest_edgeDC_id]
        # compute path cost from source edge DC to destination edge DC
        backbone_path_cost = nx.dijkstra_path_length(self.topo, source.topo, dest.topo)

        # !for now, do not consider DC-internal path cost (assume # hops from ToR to Spine are 0)
        # !when to consider, define path cost as source DC + backbone + destination DC and set self.max_path_cost accordingly
        # if isinstance(dest, LeafSpineEdgeDC) or isinstance(dest, ThreeTierEdgeDC):
        #     return backbone_path_cost + dict(nx.shortest_path_length(dest.topo))[0][len(dest.topo.nodes)-1]

        return backbone_path_cost

    # source_id: service.user_loc
    def get_least_cost_edgeDCs(self, user_loc):
        source = self.edgeDCs[user_loc]
        dict_dest_cost = dict(nx.all_pairs_dijkstra_path_length(self.topo))[source.topo]
        # remove the closest edge DC because we confirmed that all of its machines are not available
        del dict_dest_cost[source.topo]

        # try:
        #     # https://gomguard.tistory.com/137
        #     least_cost_dest_id = min(dict_dest_cost.keys(), key=(lambda k: dict_dest_cost[k]))
        #     # https://stackoverflow.com/a/17352672/5204099
        #     least_cost_dest_ids = [int(_id[len("server"):])
        #                            for _id in dict_dest_cost.keys()
        #                            if dict_dest_cost[_id] == dict_dest_cost[least_cost_dest_id]]
        #     return least_cost_dest_ids
        # except ValueError:
        #     return None

        least_cost_edgeDCs = []
        # ensure keys (topos of destination edgeDCs) are sorted according to their values (path costs) in ascending order
        for edgeDC_topo in dict_dest_cost.keys():
            edgeDC_id = str(edgeDC_topo.name).lstrip("edge")
            least_cost_edgeDCs.append(int(edgeDC_id))
        return least_cost_edgeDCs

    def add_machines(self, machine_profiles):
        for machine_profile in machine_profiles:
            machine = Machine(machine_profile)
            machine.attach(self)
            self.machines.append(machine)

    # register a service request by broker
    def add_service(self, service):
        self.services.append(service)

    def get_waiting_services(self):
        ls = []
        for service in self.services:
            if service.is_started() is False:
                ls.append(service)
        return ls

    # get list of running services
    def get_unfinished_services(self):
        ls = []
        for service in self.services:
            if service.is_started() is True and service.is_finished() is False:
                ls.append(service)
        return ls


def test():
    mec_net = MECNetwork()
    print(f"# nodes: {mec_net.topo.number_of_nodes()}, nodes: {mec_net.topo.nodes(data=True)}")
    print(f"# edges: {mec_net.topo.number_of_edges()}, edges: {mec_net.topo.edges(data=True)}")
    # mec_net.draw_topology()
    print(mec_net.max_path_cost)

    # edge_dc5 = LeafSpineEdgeDC()
    edge_dc5 = SimpleEdgeDC()
    edge_dc8 = LeafSpineEdgeDC()
    edge_dc9 = LeafSpineEdgeDC()
    dict_node_to_replace = {"server5": edge_dc5, "server8": edge_dc8, "server9": edge_dc9}
    for node in dict_node_to_replace.keys():
        edge_dc = dict_node_to_replace[node].topo
        longitude = mec_net.topo.nodes[node]["Longitude"]
        latitude = mec_net.topo.nodes[node]["Latitude"]

        # https://networkx.org/documentation/stable/tutorial.html#nodes
        # contains the nodes of edge1 as nodes of mec_net
        # mec_net.topo.add_nodes_from(edge_dc, Longitude=longitude, Latitude=latitude)
        # vs. contains edge1 as a node of mec_net
        mec_net.topo.add_node(edge_dc, Longitude=longitude, Latitude=latitude)

        # print(mec_net.topo.edges(node, data=True))
        edges = mec_net.topo.edges(node)
        for edge in edges:
            weight = mec_net.topo.edges[(edge[0], edge[1])]["weight"]
            mec_net.topo.add_edge(edge_dc, edge[1], weight=weight)
        mec_net.topo.remove_node(node)

    print(f"# nodes: {mec_net.topo.number_of_nodes()}, nodes: {mec_net.topo.nodes(data=True)}")
    print(f"# edges: {mec_net.topo.number_of_edges()}, edges: {mec_net.topo.edges(data=True)}")
    mec_net.draw_topology()

    print(nx.dijkstra_path_length(mec_net.topo, edge_dc5.topo, edge_dc8.topo))
    print(nx.dijkstra_path_length(mec_net.topo, edge_dc5.topo, edge_dc9.topo))

    # networkx.exception.NodeNotFound: Either source Graph named 'edge0' with 14 nodes and 16 edges or target {'type': 'tor'} is not in G
    # print(nx.shortest_path_length(mec_net.topo, edge_dc5.topo, edge_dc9.topo.nodes["2--7"]))
    print(nx.dijkstra_path_length(mec_net.topo, edge_dc5.topo, edge_dc9.topo)
          + dict(nx.shortest_path_length(edge_dc9.topo))[0][len(edge_dc9.topo.nodes)-1])


if __name__ == '__main__':
    os.chdir("..")
    test()
