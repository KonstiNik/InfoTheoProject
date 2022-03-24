from Compute.preprocess import PreProcess
from Compute.entropy import Entropy
from Compute.correlation import Correlation


class Evaluator:
    """
    Class for evaluating the graph
    """

    def __init__(self, nodes, edges, edge_features=None):
        self.preprocess = PreProcess(nodes, edges, edge_features)
        self.entropy = Entropy()
        self.correlation = Correlation(nodes, edges)
        self.nodes = nodes
        self.edges = edges
        self.edge_features = edge_features

    def evaluate_graph(self):
        """

        Returns
        -------
        Stationary probability of the nodes.
        """
        self.preprocess.calculate_edge_features()
        self.preprocess.calculate_stationary_node_probabilities()

        self.entropy.calculate_entropy(
            self.preprocess.edge_features,
            self.preprocess.indices_of_edges_per_node,
            self.preprocess.node_probabilities,
        )

        self.correlation.initialize(
            self.preprocess.edge_features,
            self.preprocess.node_probabilities,
            self.preprocess.number_of_edges_per_node,
            self.preprocess.indices_of_edges_per_node,
            self.entropy.value,
        )
        self.correlation.calculate_correlation()
