import numpy as np


class Entropy:
    def __init__(self):
        self.value = 0.0

    def calculate_entropy(
        self, edge_features, indices_of_edges_per_node, node_probabilities
    ):
        """
        Calculate the entropy of the given graph.

        Make use of the fact that only nodes with more than one outgoing edge contribute

        Returns
        -------

        """
        for node, indices in indices_of_edges_per_node:
            node_probability = node_probabilities[node]
            for index in indices:
                edge_probability = edge_features[index]
                self.value -= (
                    node_probability * edge_probability * np.log2(edge_probability)
                )
