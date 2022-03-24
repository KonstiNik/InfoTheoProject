from django.shortcuts import render
from Compute.evaluator import Evaluator
import networkx as nx
import matplotlib.pyplot as plt
import io
import numpy as np


def init(request):
    global G
    global NodeLabels
    global NodeLabelsS
    G = nx.DiGraph()
    NodeLabels = {}
    NodeLabelsS = {}
    global layout
    layout = "kamada"
    return indexTest(request)

def layoutf(layout ="kamada"):
    if(len(G.nodes)  == 0):
        return []
    if(layout == "kamada"):
        return nx.kamada_kawai_layout(G)
    if(layout == "spring"):
        return nx.spring_layout(G)
    if(layout == "planar"):
        return nx.planar_layout(G)
    if(layout == "circular"):
        return nx.circular_layout(G)
    if(layout == "random"):
        return nx.random_layout(G)
    else:
        return nx.kamada_kawai_layout(G)

def plotComplexity(agent):
    sequence_length = np.arange(
        1, len(agent.correlation.correlation_complexity_list) + 1
    )
    fig, axs = plt.subplots(2)
    axs[0].plot(sequence_length, agent.correlation.block_entropy, 'o-')
    axs[0].tick_params("x", labelbottom=False)
    axs[0].set_ylabel("Block Entropy $S$")
    axs[1].set_ylabel("Correlation Complexity $\eta$")
    axs[1].set_xlabel("Sequence length n")
    axs[1].plot(sequence_length, agent.correlation.correlation_complexity_list, 'o-')
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.savefig("static/Complexity.png", format="PNG")
    plt.close()

def draw():
    pos = layoutf(layout)
    color_map = []
    for node in G:
        if NodeLabels[node] == "1":
            color_map.append('red')
        else:
            color_map.append('green')
    node_labels = NodeLabelsS
    if(len(G.nodes) > 0):
        shifted_pos = {k: [v[0], v[1]] for k, v in pos.items()}
        node_label_handles = nx.draw_networkx_labels(G, pos=shifted_pos,
                                                   labels=node_labels)
    nx.draw_networkx(G, pos=pos,node_size=540, with_labels=False, arrowsize=35, node_color=color_map)
    ax = plt.gca()
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.savefig("static/Graph.png", format="PNG")
    plt.close()

    plt.figure()
    plt.savefig("static/Complexity.png", format="PNG")
    plt.close()

def draw2():
    pos = nx.random_layout(G)
    nx.draw(G, pos = pos, with_labels = True)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    #image_bytes = buf.getvalue().decode('utf-8')
    plt.savefig("static/images/Graph.png", format="PNG")
    plt.close()


def drawMethod(request):
    global layout
    layout = request.POST['layout']
    draw()
    return render(request, "indexTest.html", context={})

def loadSide(request):
    draw()
    return render(request, "indexTest.html", context={})

def addEdge(request):
    vertex1 = request.POST['from']
    vertex2 = request.POST['to']
    if((not NodeLabels.__contains__(vertex1)) or (not NodeLabels.__contains__(vertex2))):
        return render(request, "error.html", context={'error': "At least one of the vertices does not exist"})
    G.add_edge(vertex1, vertex2)
    draw()
    return render(request, "indexTest.html", context={})

def removeEdge(request):
    vertex1 = request.POST['from']
    vertex2 = request.POST['to']
    try:
        G.remove_edge(vertex1,vertex2)
    except: return render(request, "error.html", context={'error':"Deletion of a nonexistant edge"})

    draw()
    return render(request, "indexTest.html", context={})

def removeVertex(request):
    name = request.POST['name']
    try:
        G.remove_node(name)
    except: return render(request, "error.html", context={'error':"Deletion of a nonexistant vertex"})
    NodeLabels[name] = None
    NodeLabelsS.__delitem__(name)
    draw()
    return render(request, "indexTest.html", context={})

def index(request):
    name = request.POST['name']
    value = request.POST['output']
    G.add_node(name)
    NodeLabels[name] = value
    NodeLabelsS[name] = name + ", " + value
    draw()
    return render(request, "indexTest.html", context={})

def indexTest(request):
    draw()
    return render(request, "indexTest.html", context={})


def convertToGraph():
    mappi = {}
    i = 0
    for v in G.nodes:
        mappi[v] = i
        i = i + 1
    vertices = list(map(lambda x: [int(NodeLabels[x])], G.nodes))
    edges = list(map(lambda e: (mappi[e[0]],mappi[e[1]]), G.edges))
    first = list(map(lambda a: a[0], edges))
    second = list(map(lambda a: a[1], edges))
    edges = [first, second]
    return vertices, edges



def compute(request):
    x, edge_index = convertToGraph()
    agent = Evaluator(nodes=x, edges=edge_index)
    try: agent.evaluate_graph()
    except: return render(request, "error.html", context={'error':"Non-valid graph"})
    entropy = agent.entropy.value
    distribution = agent.preprocess.node_probabilities
    length = agent.correlation.correlation_length
    complexity = agent.correlation.correlation_complexity
    draw()
    plotComplexity(agent)
    return render(request, "indexTest.html", context={'name': 'abc', 'title': 'KonstiKannNix','entropy': entropy,'distribution': distribution, 'length': length, 'complexity': complexity })
