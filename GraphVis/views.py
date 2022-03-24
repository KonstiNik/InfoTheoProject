from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from pyvis.network import Network
from Compute.evaluator import Evaluator
import networkx as nx
import matplotlib.pyplot as plt
import io
import numpy as np

global context_data;

def init(request):
    global G
    global NodeLabels
    global NodeLabelsS
    G = nx.DiGraph()
    NodeLabels = {}
    NodeLabelsS = {}
    global layout
    layout= "kamada"
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
    else:
        return nx.kamada_kawai_layout(G)

def draw():
    print("hi")
    pos = layoutf(layout)
    color_map = []
    for node in G:
        if NodeLabels[node] == "1":
            color_map.append('red')
        else:
            color_map.append('green')
    node_labels = NodeLabelsS
    # draw the graph

    if(len(G.nodes) > 0):
        shifted_pos = {k: [v[0], v[1]] for k, v in pos.items()}
        node_label_handles = nx.draw_networkx_labels(G, pos=shifted_pos,
                                                   labels=node_labels)
    nx.draw_networkx(G, pos=pos,node_size=540, with_labels=False, arrowsize=35, node_color=color_map)

    # draw the custom node labels
    # add a white bounding box behind the node labels
   # [label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in
   #  node_label_handles.values()]
    ax = plt.gca()
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    # add the custom egde labels
    #nx.draw_networkx_edge_labels(G, pos=pos)
    buf = io.BytesIO()
    #plt.figure(figsize=(12,12))
    plt.savefig(buf, format='svg', bbox_inches='tight')
    #image_bytes = buf.getvalue().decode('utf-8')
    plt.savefig("static/Funny.png", format="PNG")
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
    return render(request, "indexTest.html", context={'name': 'nc', 'title': 'dasd'})

def loadSide(request):
    draw()
    return render(request, "indexTest.html", context={'name': 'nc', 'title': 'dasd'})

def addEdge(request):
    vertex1 = request.POST['from']
    vertex2 = request.POST['to']
    edgeVal=request.POST['prob']
    if((not NodeLabels.__contains__(vertex1)) or (not NodeLabels.__contains__(vertex2))):
        return render(request, "error.html", context={'error': "At least one of the vertices does not exist"})
    G.add_edge(vertex1, vertex2)
    draw()
    return render(request, "indexTest.html", context={'name': 'nc', 'title': 'dasd'})

def removeEdge(request):
    vertex1 = request.POST['from']
    vertex2 = request.POST['to']
    try:
        G.remove_edge(vertex1,vertex2)
    except: return render(request, "error.html", context={'error':"Deletion of a nonexistant edge"})

    draw()
    return render(request, "indexTest.html", context={'name': 'nc', 'title': 'dasd'})

def removeVertex(request):
    name = request.POST['name']
    try:
        G.remove_node(name)
    except: return render(request, "error.html", context={'error':"Deletion of a nonexistant vertex"})
    NodeLabels[name] = None
    NodeLabelsS.__delitem__(name)
    draw()
    return render(request, "indexTest.html", context={'name': name, 'title': 'dasd'})

def index(request):
    name = request.POST['name']
    value = request.POST['output']
    G.add_node(name)
    NodeLabels[name] = value
    NodeLabelsS[name] = name + ", " + value
    draw()
    return render(request, "indexTest.html", context={'name': name, 'title': 'dasd'})

def indexTest(request):
    draw()
    return render(request, "indexTest.html", context={'name': 'abc', 'title': 'KonstiKannNix'})


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
    agent.evaluate_graph()
    entropy = agent.entropy.value
    distribution = agent.preprocess.node_probabilities
    length = agent.correlation.correlation_length
    complexity = agent.correlation.correlation_complexity
    draw()
    return render(request, "indexTest.html", context={'name': 'abc', 'title': 'KonstiKannNix','entropy': entropy,'distribution': distribution, 'length': length, 'complexity': complexity })



def search(request):
    G = nx.DiGraph(day="friday")
    G.add_node(1, time="5pm")
    G.add_node(2, time="3pm")
    G.add_edge(1, 2, weight=4.7)
    nx.draw(G)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    image_bytes = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close()
    html = ("<H1>%s</H1>", image_bytes)
    return HttpResponse(html)

def index4(request, context_data=None):
    G = nx.complete_graph(5)
    nx.draw(G)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    image_bytes = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close()

    context_data['my_chart'] = image_bytes
    return render(request,   "index.html", context={'hello':'world'})
# Create your views here.

def index2(request):
    G = nx.DiGraph(day = "friday")
    G.add_node(1, time="5pm")
    G.add_node(2, time = "3pm")
    G.add_edge(1, 2, weight=4.7)
    nx.draw(G)
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    image_bytes = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close()
    now = 2
    html = "<html><body>Python skills = infinity - epsilon %s.</body></html>" % image_bytes
    with open('templates/index.html', 'r') as file:
        data = file.read().replace('\n', '')
    data = data + html
    return HttpResponse(data)

def index5(request):
    G = nx.DiGraph(day = "friday")
    G.add_node(1, time="5pm")
    G.add_node(2, time = "3pm")
    G.add_edge(1, 2, weight=4.7)

    g = Network(height=800, width=800, notebook= True)
    g.barnes_hut()
    g.from_nx(G)
    a = g.show("ex.html")
    html =  "<html><body>Python skills = infinity - epsilon %s.</body></html>" % a
    return HttpResponse(html)

def index12(request):

    json_object = {'key': "value"}
    return JsonResponse(json_object)