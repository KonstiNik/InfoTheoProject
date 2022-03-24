import numpy as np


class PreProcess:
    """
    Class for preprocessing of the graph.
    """

    def __init__(self, nodes, edges, edge_features):
        """
        Init of the Evaluator class

        Parameters
        ----------
        nodes : list
                nodes of the graph with respective labels
        edges : list
                edges of the graph
        edge_features : list
                probabilities of the edges
        """
        self.nodes = nodes
        self.edges = edges
        self.edge_features = edge_features

        self.number_of_edges_per_node = None
        self.indices_of_edges_per_node = None

        self.flow = None
        self.node_probabilities = None

    def calculate_edge_features(self):
        """
        Calculate edge features by assuming each edge to be equally probable,
        if not already given

        Returns
        -------
        matrix with shape [num_edges, num_edge_features]
        """
        self._get_node_information()

        if self.edge_features is None:
            # Create list of feature probabilities
            self.edge_features = [1] * len(self.edges[0])

            for ind, node in enumerate(self.number_of_edges_per_node):
                probability = 1 / node[1]
                for position in self.indices_of_edges_per_node[ind][1]:
                    self.edge_features[position] *= probability

    def _get_node_information(self):
        """
        Get information which nodes have more than one outgoing edge and the respective
        indices of the position in self.edge_features.
        """
        multi_edge_nodes = self.get_nodes_with_multiple_edges(
            edges=self.edges, outgoing=True
        )
        (
            self.number_of_edges_per_node,
            self.indices_of_edges_per_node,
        ) = self.get_number_of_edges_per_node(
            edges=self.edges, multi_edge_nodes=multi_edge_nodes, outgoing=True
        )

    @staticmethod
    def get_number_of_edges_per_node(edges, multi_edge_nodes, outgoing=True):
        """
        Get the number of edges per node

        Parameters
        ----------
        edges : iterable
                list of edges
        multi_edge_nodes : iterable
                list of which nodes have multiple outgoing edges
        outgoing : bool
                If True, get number of outgoing edges per node
                If False, get number of incoming edges per node

        Returns
        -------
        List of number of edges per node, where the sublists contain the index of the
        node in the first entry and the number of edges in the second.

        List of indices of edges per node, where the sublists contain the index of the
        node in first entry and the second entry is a list of the indices of the edges.
        """
        number_of_edges_per_node = []
        indices_of_edges_per_node = []
        index = 0
        if outgoing is False:
            index = 1
        for multi_edge_node in multi_edge_nodes:
            indices_multi_edge_nodes = [
                idx for idx, node in enumerate(edges[index]) if node == multi_edge_node
            ]
            number_of_edges_per_node.append(
                [multi_edge_node, len(indices_multi_edge_nodes)]
            )
            indices_of_edges_per_node.append(
                [multi_edge_node, indices_multi_edge_nodes]
            )
        return number_of_edges_per_node, indices_of_edges_per_node

    @staticmethod
    def get_nodes_with_multiple_edges(edges, outgoing: bool = True) -> list:
        """
        Get all nodes with more than one outgoing / incoming edge

        Parameters
        ----------
        edges : iterable
                list of edges
        outgoing : bool
                If True, look for multiple outgoing edges.
                If False, look for multiple incoming edges.
        Returns
        -------
        List of nodes with more than one outgoing / incoming edge
        """
        seen = set()
        duplicated_nodes = []
        index = 0
        if outgoing is False:
            index = 1
        for node in edges[index]:
            if node in seen:
                duplicated_nodes.append(node)
            else:
                seen.add(node)
        return duplicated_nodes

    @staticmethod
    def calculate_transition_matrix(
        nodes,
        edges,
        edge_features,
        alphabet,
        reverse=False,
    ):
        """
        Calculate matrix of transition probabilities between nodes.

        Parameters
        ----------
        nodes : iterable
                nodes of a graph
        edges : iterable
                edges of a graph
        edge_features : iterable
                transition probabilities of the respective edge
        reverse : bool
                if True, look for multiple outgoing edges.
                if False, look for multiple incoming edges.
        alphabet : list
                Symbol alphabet

        Returns
        -------
        Numpy array consisting of l nxn transition matrices, when l is the length of
        the alphabet and n the number of nodes of the graph.
        Each transition matrix contains all transition probabilities to a symbol.
        """
        if alphabet is None:
            alphabet = [0, 1]
        nodes = np.squeeze(nodes, axis=-1)
        transition_matrix = []
        index = 1
        if reverse is True:
            index = 0
        for symbol in alphabet:
            node_indices = np.argwhere(nodes == symbol)
            edge_indices = [
                np.argwhere(node == edges[index]).squeeze() for node in node_indices
            ]
            edge_indices = np.hstack(edge_indices)
            transition = np.zeros(shape=(len(nodes), len(nodes)))
            for edge_index in edge_indices:
                i, j = edges[1 - index][edge_index], edges[index][edge_index]
                transition[i, j] += edge_features[edge_index]
            transition_matrix.append(transition.tolist())
        return np.array(transition_matrix)

    @staticmethod
    def categorize_node_probabilities(nodes, node_probabilities, alphabet):
        """
        Assigning a node probabilities to the respective symbol and put them in lists.

        Parameters
        ----------
        nodes : list
                list of nodes with symbols
        node_probabilities : list
                stationary node probabilities
        alphabet : set
                alphabet of the graph

        Returns
        -------
        List of node probabilities seperated by their respective symbol.
        Lists are ordered in terms of the alphabet.
        """
        start_probabilities = []
        for symbol in alphabet:
            idx = np.argwhere(np.array(nodes).squeeze() == symbol)
            array = np.zeros_like(node_probabilities)
            array[idx] += node_probabilities[idx]
            start_probabilities.append(array.tolist())
        return start_probabilities

    def calculate_stationary_node_probabilities(self):
        """
        Calculate the stationary probabilities of each node in the graph
        """
        # get alphabet
        alphabet = self.get_alphabet(self.nodes)
        # calculate transition matrix
        transition_matrix = self.calculate_transition_matrix(
            self.nodes, self.edges, self.edge_features, alphabet, reverse=False
        )
        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.sum(axis=0).T)
        close_to_1_idx = np.isclose(eigenvalues, 1)
        # get eigenvector
        target_eigenvector = eigenvectors[:, close_to_1_idx]
        target_eigenvector = target_eigenvector[:, 0]
        # normalize eigenvector
        stationary_distribution = target_eigenvector / target_eigenvector.sum()
        self.node_probabilities = np.real(stationary_distribution)

    @staticmethod
    def get_alphabet(nodes):
        """
        Get the alphabet of the graph

        Parameters
        ----------
        nodes : set
                nodes with corresponding symbols of the graph

        Returns
        -------
        List of alphabet
        """
        nodes = np.array(nodes)
        alphabet = sorted(set(nodes.squeeze(axis=-1)))
        return list(alphabet)
