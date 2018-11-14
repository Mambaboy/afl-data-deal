import os 
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

from networkx.drawing.nx_agraph import graphviz_layout

def main():
    datadir="/dev/shm/collect-fuzzed/queue"
    G=nx.DiGraph()
    inputs = glob.glob(datadir+"/id*")
    inputs.sort()
    print len(inputs)
    
    node_size= np.full(len(inputs),3)
    node_color= np.full(len(inputs),"r")
    
    for input in inputs[0:]:
        input = os.path.basename(input)
        if "orig" in input:
            curid=int(input[3:9])
            node_color[curid]="b"
            G.add_node(curid)
            continue
        curid=int(input[3:9])
        srcid=int(input[14:20])
        op=input[20:]
        if node_size[srcid]<200:
            node_size[srcid]+=20

        if srcid==0:
            node_color[curid]="y"
            node_size[curid]=200
            #continue
            pass
        G.add_edges_from( [(srcid,curid,{"op": op})] )
    print ("nodes :",len(G.nodes) )
    print ("edges :",len(G.edges) )
 
    plt.figure(figsize=(25,25))

    pos = graphviz_layout(G, prog='dot') 
    nx.draw(G, pos, with_labels=True, arrows=False)

    #nx.draw_spring(G, node_size=node_size, node_color=node_color)
    plt.savefig("test.png")


if __name__ == "__main__":
    main();
