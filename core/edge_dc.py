import networkx as nx
from abc import *
from config import cfg
from core.machine import MachineProfile, Machine


class EdgeDC(metaclass=ABCMeta):
    global_index = -1

    def __init__(self):
        self.id = EdgeDC.get_global_id()
        self.topo = self.create_topology()

    @classmethod
    def get_global_id(cls):
        cls.global_index += 1
        return cls.global_index

    @abstractmethod
    def create_topology(self):
        pass

    @abstractmethod
    def create_machines(self, mec_net):
        pass


# represents an edge dc with a single machine (mapped to the simple node)
class SimpleEdgeDC(EdgeDC):

    def __init__(self):
        super().__init__()

    def create_topology(self):
        name = "cloud" + str(self.id)
        G = nx.Graph(name=name)
        G.add_nodes_from([(0, {"type": "simple"})])
        # print(f"# nodes: {G.number_of_nodes()}, nodes: {G.nodes(data=True)}")
        # print(f"# edges: {G.number_of_edges()}, edges: {G.edges(data=True)}")
        return G

    def create_machines(self, mec_net):
        # !add edgeDC_id to MachineProfile (not to impede backward compatibility).
        # !a machine can access its edgeDC via its mec_net.edgeDCs with its machine_profile.edgeDC_id
        machine_profiles = [MachineProfile(*list(cfg["simple_edge"]["machine_profile"].values()), edgeDC_id=self.id)]
        mec_net.add_machines(machine_profiles)


class LeafSpineEdgeDC(EdgeDC):

    def __init__(self, num_spines=2, num_leaves=4, leaf_fanout=2):
        self.num_spines = num_spines
        self.num_leaves = num_leaves
        self.leaf_fanout = leaf_fanout
        super().__init__()

    def create_topology(self):
        name = "edge" + str(self.id)
        G = nx.Graph(name=name)

        # spine_node_list = [(str(self.id) + str(i) + "--", {"type": "spine"}) for i in range(self.num_spines)]
        spine_node_list = [(i, {"type": "spine"}) for i in range(self.num_spines)]
        G.add_nodes_from(spine_node_list)

        leaf_node_list = [(self.num_spines + i, {"type": "leaf"}) for i in range(self.num_leaves)]
        G.add_nodes_from(leaf_node_list)

        # add links between spines and leaves
        for i in range(self.num_spines):
            for j in range(self.num_leaves):
                G.add_edge(spine_node_list[i][0], leaf_node_list[j][0])

        num_tors = self.num_leaves * self.leaf_fanout
        tor_node_list = [(self.num_spines + self.num_leaves + i, {"type": "tor"}) for i in range(num_tors)]
        G.add_nodes_from(tor_node_list)

        # add links between leaves and tors
        j = 0
        for i in range(self.num_leaves):
            G.add_edge(leaf_node_list[i][0], tor_node_list[j][0])
            G.add_edge(leaf_node_list[i][0], tor_node_list[j+1][0])
            j += self.leaf_fanout

        # print(f"# nodes: {G.number_of_nodes()}, nodes: {G.nodes(data=True)}")
        # print(f"# edges: {G.number_of_edges()}, edges: {G.edges(data=True)}")
        # print(dict(nx.all_pairs_dijkstra(G)))
        # print(dict(nx.all_pairs_dijkstra_path(G)))
        # print(dict(nx.all_pairs_dijkstra_path_length(G)))

        return G

    def create_machines(self, mec_net):
        num_machines = cfg["leaf_spine_edge"]["num_leaves"] * cfg["leaf_spine_edge"]["num_fanout"]
        # !add edgeDC_id to MachineProfile (not to impede backward compatibility).
        # !a machine can access its edgeDC via its mec_net.edgeDCs with its machine_profile.edgeDC_id
        machine_profiles = [MachineProfile(*list(cfg["leaf_spine_edge"]["machine_profile"].values()), edgeDC_id=self.id)
                            for _ in range(num_machines)]
        mec_net.add_machines(machine_profiles)


# TODO:
class ThreeTierEdgeDC(EdgeDC):

    def __init__(self):
        super().__init__()

    def create_topology(self):
        pass

    def create_machines(self, mec_net):
        pass


def test():
    edge0 = SimpleEdgeDC()
    edge1 = LeafSpineEdgeDC()
    # edge.draw_topology()


if __name__ == '__main__':
    test()
