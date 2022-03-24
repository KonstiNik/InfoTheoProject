from Compute.evaluator import Evaluator
import numpy as np
import matplotlib.pyplot as plt


### This is an example script of how to execute the algorithm in python ###
### withou using the GUI ###

if __name__ == "__main__":

    x = [[1], [0], [0], [1]]
    edge_index = [[0, 1, 1, 2, 3],
                  [1, 2, 3, 0, 0]]
    # edge_features = [1, 0.5, 0.5, 1, 1]

    # x = [[0], [0], [1]]
    # edge_index = [[0, 1, 1, 2, 2],
    #               [1, 0, 2, 1, 2]]

    # x = [[1], [0]]
    # edge_index = [[0, 1, 1, 0],
    #               [1, 0, 1, 0]]

    # x = [[1], [0]]
    # edge_index = [[0, 1],
    #               [1, 0]]

    # x = [[0]]
    # edge_index = [[0],
    #               [0]]

    # x = [[1], [0], [0]]
    # edge_index = [[0, 1, 1, 2, 0],
    #               [1, 0, 2, 0, 0]]

    # x = [[1], [1], [0], [0]]
    # edge_index = [[0, 1, 1, 2, 3, 3],
    #               [1, 1, 2, 3, 0, 3]]

    # x = [[0], [1], [0], [0]]
    # edge_index = [[0, 1, 2, 0, 3],
    #               [1, 2, 0, 3, 2]]

    # x = [[1], [0], [0]]
    # edge_index = [[0, 2, 1, 2],
    #               [1, 0, 2, 1]]

    # x = [[1], [0], [0], [1], [0]]
    # edge_index = [[0, 2, 1, 2, 1, 3, 4, 4],
    #               [1, 0, 2, 1, 3, 4, 4, 2]]

    # x = [[1], [0]]
    # edge_index = [[0, 1, 1],
    #               [1, 0, 1]]

    #### Execution ####
    agent = Evaluator(nodes=x, edges=edge_index)
    agent.evaluate_graph()
    print(
        f" Entropy = {agent.entropy.value} \n",
        f"Node Probabilities: {agent.preprocess.node_probabilities} \n",
        f"Correlation Length = {agent.correlation.correlation_length} \n",
        f"Correlation Complexity {agent.correlation.correlation_complexity}",
    )

    #### Plot ####
    sequence_length = np.arange(
        1, len(agent.correlation.correlation_complexity_list) + 1
    )
    # fig, axs = plt.subplots(2, 1)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(sequence_length, agent.correlation.block_entropy, 'o-')
    axs[0].tick_params("x", labelbottom=False)
    axs[0].set_ylabel("Block Entropy $S$")
    axs[1].set_ylabel("Correlation Complexity $\eta$")
    axs[1].set_xlabel("Sequence length n")
    axs[1].plot(sequence_length, agent.correlation.correlation_complexity_list, 'o-')

    plt.show()
