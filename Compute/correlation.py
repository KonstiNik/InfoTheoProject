import numpy as np
from Compute.preprocess import PreProcess


class Correlation:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self.edge_features = None
        self.node_probabilities = None
        self.number_of_edges_per_node = None
        self.indices_of_edges_per_node = None
        self.entropy = None
        self.alphabet = None

        self.transition_matrix = None

        self.preprocess = None
        self.break_condition = False
        self.sequence_length = 1

        self.state: np.array = None
        self.probabilities_x1: list = []
        self.probabilities_x2: list = []

        self.conditional_probabilities_x1: list = []
        self.conditional_probabilities_x2: list = []

        self.correlation_information: list = []

        self.correlation: list = []
        self.block_entropy = []

        self.correlation_complexity = None
        self.correlation_complexity_list = None
        self.correlation_length = None

        self.reverse = False

    def initialize(
        self,
        edge_features,
        node_probabilities,
        number_of_edges_per_node,
        indices_of_edges_per_node,
        entropy,
    ):
        self.edge_features = edge_features
        self.node_probabilities = node_probabilities
        self.number_of_edges_per_node = number_of_edges_per_node
        self.indices_of_edges_per_node = indices_of_edges_per_node
        self.entropy = entropy

    def set_up_preprocess(self):
        """
        Setup Preprocess class to access the methods
        """
        self.preprocess = PreProcess(self.nodes, self.edges, self.edge_features)

    def calculate_inverse_transition_matrix(self):
        """
        Calculate the inverse transition matrix

        Returns
        -------
        Inverse transition matrix
        """
        self.transition_matrix = self.preprocess.calculate_transition_matrix(
            nodes=self.nodes,
            edges=self.edges,
            edge_features=self.edge_features,
            alphabet=self.alphabet,
            reverse=self.reverse,
        )

    def get_start_probabilities(self):
        """
        Assigning a node probabilities to the respective symbol and put them in lists.
        """
        start_probabilities = self.preprocess.categorize_node_probabilities(
            self.nodes, self.node_probabilities, self.alphabet
        )
        self.probabilities_x1.append(
            np.array(start_probabilities).sum(axis=-1).tolist()
        )
        self.state = np.array(start_probabilities)

    def get_alphabet(self):
        """
        Get the alphabet of the graph
        """
        self.alphabet = self.preprocess.get_alphabet(self.nodes)

    def generate_new_symbol(self):
        """
        Calculate probabilities of the next symbol

        Returns
        -------
        Adds all new possible states to the probabilities list
        """
        old_probabilities = np.array(self.state)
        new_probabilities = np.dot(old_probabilities, self.transition_matrix)
        self.state = new_probabilities

    def calculate_conditional_probabilities(self):
        """
        Calculate conditional probabilities of reading a new symbol.
        """

        self.conditional_probabilities_x1.append(
            self.calculate_cond_probabilities(self.probabilities_x1)
        )
        self.conditional_probabilities_x2.append(
            self.calculate_cond_probabilities(self.probabilities_x2)
        )

    @staticmethod
    def calculate_cond_probabilities(probabilities):
        """
        Calculate the conditional probabilities of a given probability input.
        The input contains lists of probabilities of subsequent symbol sequences.

        Parameters
        ----------
        probabilities : iterative
                probabilities of subsequent symbol sequences

        Returns
        -------
        List of conditional probabilities
        """
        if len(probabilities) == 1:
            return probabilities[0]
        else:
            old_sequence = np.expand_dims(probabilities[-2], axis=-1)
            new_sequence = np.array(probabilities[-1])
            cond_prob = np.divide(
                new_sequence,
                old_sequence,
                out=np.zeros_like(new_sequence),
                where=old_sequence != 0,
            )
            return cond_prob.tolist()

    def calculate_probabilities_x1(self):
        """
        Calculate the probability of a symbol sequence starting from the first symbol
        """
        self.probabilities_x1.append(self.state.sum(axis=-1).tolist())

    def calculate_probabilities_x2(self):
        """
        Calculate the probability of a symbol sequence starting from the second symbol
        """
        probabilities_x1 = np.array(self.probabilities_x1[-1])
        probabilities_x2 = np.array(np.split(probabilities_x1, len(self.alphabet))).sum(
            axis=0
        )
        self.probabilities_x2.append(probabilities_x2.tolist())

    def calculate_probabilities(self):
        """
        Calculate the probabilities for a symbol sequence and append it to the list
        """
        self.calculate_probabilities_x1()
        self.calculate_probabilities_x2()

    def calculate_block_entropy(self):
        """
        Calculate the block entropy of a given symbol sequence

        Returns
        -------
        Appends block entropy to the list
        """
        probabilities_x1 = np.array(self.probabilities_x1[-1]).flatten()
        entropy = self.calculate_entropy(probabilities_x1)
        self.block_entropy.append(entropy)

    @staticmethod
    def calculate_entropy(distribution):
        """
        Calculate the entropy of a given probability distribution

        Parameters
        ----------
        distribution : np.array
                input distribution to calculate the entropy from

        Returns
        -------
        Entropy of the given distribution
        """
        distribution = distribution[distribution != 0]  # remove all zeros
        return np.sum(-distribution * np.log2(distribution))

    def calculate_correlation_information(self):
        """
        Calculate correlation information of a given symbol sequence.
        Calculate k1 separately, since it has no preceding sequence.
        """
        if len(self.correlation_information) == 0:
            self.calculate_k1()
        if len(self.correlation_information) >= 0:
            self.calculate_kn()

    def calculate_k1(self):
        """
        Calculate first coefficient of the correlation information.
        This is done based on the fact that
            k_1 = log_n(n) - S_1
        with n being the length of the alphabet and S_1 the block entropy of sequences
        with length 1.
        """
        self.correlation_information.append(1 - self.block_entropy[0])

    def calculate_kn(self):
        """
        Calculate the correlation information of any symbol sequence bigger than 1
        """
        prob = np.expand_dims(self.probabilities_x1[-2], axis=0)
        cond_prob_x1 = np.array(self.conditional_probabilities_x1[-1])
        cond_prob_x2 = np.array(self.conditional_probabilities_x2[-1])

        result = []
        for array in np.split(cond_prob_x1, len(self.alphabet)):
            division = np.divide(
                array, cond_prob_x2, out=np.zeros_like(array), where=cond_prob_x2 != 0
            )
            division = np.where(division > 0.0, division, 1)
            log = np.log2(division)
            result.append(log.tolist())

        final_matrix = (
            np.expand_dims(np.squeeze(prob, axis=0), axis=-1)
            * cond_prob_x1
            * np.array(result).squeeze()
        )
        self.correlation_information.append(final_matrix.sum())

    def calculate_correlation_length(self):
        """
        Calculate the correlation length of a given graph based on the correaltion
        information.

        Returns
        -------

        """
        k = np.array(self.correlation_information)[::-1]
        index = (k != 0).argmax(axis=0)
        correlation_length = len(k) - index
        if correlation_length == len(k):
            self.correlation_length = "inf"
        else:
            self.correlation_length = correlation_length

    def calculate_correlation_complexity(self):
        """
        Calculate the correlation complexity

        In case of a finite correlation length, the calculation is straight forward.
        It will be calculated in two ways and asserted.

        In case of an infinite correlation length, the complexity is calculated by
        two terms, of which we do not calculate the second. The second term is always
        negative, since we approximate the correlation complexity to be smaller equal
        the first term.
        """
        eta_1 = self.calculate_corr_com_from_correlation_information()
        eta_2 = self.calculate_corr_com_from_block_entropy()
        if self.correlation_length == "inf":
            prob = np.where(self.node_probabilities > 0.0, self.node_probabilities, 1)
            self.correlation_complexity = f"≤ {(-prob * np.log2(prob)).sum()}"
            self.correlation_complexity_list = eta_2
            print("calculation successful")
        else:
            if np.allclose(eta_1, eta_2[-1]) is True:
                print("calculation successful")
                self.correlation_complexity = f"= {eta_1}"
                self.correlation_complexity_list = eta_2
            else:
                print("something went wrong")

    def calculate_corr_com_from_correlation_information(self):
        """
        Calculate the correlation complexity based on the correlation information

        Returns
        -------
        Correlation complexity
        """
        k = np.array(self.correlation_information)
        weight = np.arange(0, len(k))
        return (weight * k).sum()

    def calculate_corr_com_from_block_entropy(self):
        """
        Calculate the correlation complexity using the block entropy

        Returns
        -------
        List of block correlation complexities for each sequence length
        """
        correlation_complexity = []
        for index, block_entropy in enumerate(self.block_entropy):
            correlation_complexity.append(block_entropy - (index + 1) * self.entropy)
        return correlation_complexity

    def check_break_condition(self):
        """
        Check if break conditions are fulfilled.
        Todo: Implement the second condition defined by correlation length
        """
        if self.sequence_length > 2 * len(self.nodes):
            self.break_condition = True

    def iterate_over_sequences(self):
        """
        Iterate over an increasing number of symbol sequences.
        Break when the correlation is zero or the the graph is fully explored.
        """
        while self.break_condition is False:
            self.generate_new_symbol()
            self.sequence_length += 1

            self.calculate_probabilities()
            self.calculate_block_entropy()
            self.calculate_conditional_probabilities()
            self.calculate_correlation_information()
            self.check_break_condition()

    def calculate_correlation(self):
        """
        Calculate correlation properties of a given input graph.
        """
        self.set_up_preprocess()
        self.get_alphabet()
        self.calculate_inverse_transition_matrix()
        self.get_start_probabilities()
        self.calculate_block_entropy()
        self.iterate_over_sequences()
        self.calculate_correlation_length()
        self.calculate_correlation_complexity()
