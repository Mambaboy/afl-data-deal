import os 
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

def main():
    datadir="/dev/shm/collect-fuzzed/queue"
    G=nx.DiGraph()
    inputs = glob.glob(datadir+"/id*")
    inputs.sort()
    print len(inputs)
    
    node_size= np.full(len(inputs),3)
    
    for input in inputs[0:]:
        input = os.path.basename(input)
        if "orig" in input:
            curid=int(input[3:9])
            G.add_node(curid)
            continue
        curid=int(input[3:9])
        srcid=int(input[14:20])
        op=input[20:]
        if node_size[srcid]<200:
            node_size[srcid]+=20

        if srcid==0:
            #continue
            pass
        G.add_edges_from( [(srcid,curid,{"op": op})] )
    print ("nodes :",len(G.nodes) )
    print ("edges :",len(G.edges) )
    
    plt.figure(figsize=(25,25))

    nx.draw(G, node_size=node_size)
    plt.savefig("test.png")


if __name__ == "__main__":
    main();
