###script for calculating network measures and storing in csv.

###################################import required modules######################################


import numpy as np
import pandas as pd
import scipy as sc
import networkx as nx
import matplotlib.pyplot as plt
import os 
from natsort import natsorted
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize']=24
mpl.rcParams['figure.subplot.right']=0.95
mpl.rcParams['figure.subplot.top']=0.95
mpl.rcParams['figure.subplot.bottom']=0.15
mpl.rcParams['figure.subplot.wspace']=0.0 
mpl.rcParams['figure.subplot.hspace']=0.0 
label_size = 20
#mpl.rcParams['legend.labelsize'] = label_size 
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
##

import glob

topDir = '/Users/carmenlee/Documents/NCSU/DATA/Networks/PC10021/' #path to image
#topDir = '/oit-rs/rsstu/dmref-networks/ConfigurationLibrary/ConfigLib/DATA/PC10021/'
post = '*_Adj.csv'  #the string before and after the numbers you want to change, must be unique to the directory
fs = glob.glob(topDir+post, recursive=True)
#files = 
files= natsorted(fs)
#print(files)

xy = '*xy.csv'
fxy = glob.glob(topDir+xy, recursive=True)
filesxy=natsorted(fxy)

#r = '/Users/carmenlee/Documents/NCSU/DATA/Networks/RmatB.csv'
#rmat = np.genfromtxt(r, delimiter = ',', skip_header = 1)

#figure = plt.figure()
#ax = figure.add_subplot()
#ax.plot(rmat.T)

def find_corners(data):
    """
    Find the four corners of a roughly square-shaped set of points.

    Args:
        data (numpy.ndarray): Array of shape (n, 2) containing the (x, y) coordinates of the points.

    Returns:
        numpy.ndarray: Array of shape (4, 2) containing the coordinates of the four corners.
    """
    # Find the minimum and maximum x and y values
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    # Group into corners
    corners = np.array([
        [x_min, y_min],  # Bottom-left
        [x_max, y_min],  # Bottom-right
        [x_min, y_max],  # Top-left
        [x_max, y_max],  # Top-right
    ])
    closest_points = []
    closest_idex = []
    for corner in corners:
        distances = np.linalg.norm(data - corner, axis=1)  # Euclidean distance to all points
        closest_idx = np.argmin(distances)  # Index of the closest point
        closest_idex.append(closest_idx)
        closest_points.append(data[closest_idx])  # Append the closest point

    return corners, np.array(closest_points), closest_idex
    

#Reff, Fiedler
measurenames = ["Nodes", "Betweenness Centrality", "Degree Centrality", "Closeness Centrality","Clustering", "Shortest path A", "Shortest path B", "Average Node Degree", "Edgeweights", "Entropy", "ReffA", "ReffB"]
networkmeasures = np.zeros(( len(files), 12))
for f in range(len(files)):
#for f in range(60,62):
    data = np.genfromtxt(files[f], delimiter = ',', dtype=int)
    positions = np.genfromtxt(filesxy[f], delimiter =',')
    corners, closestcorner, closestidx = find_corners(positions)
    #print(closestcorner, closestidx)
    
    A = [closestidx[0], closestidx[3]]
    B = closestidx[1:3]

    '''plt.scatter(positions[:, 0], positions[:, 1], label="Data points")
    plt.scatter(positions[closestidx, 0], positions[closestidx, 1], label="corner")
    plt.scatter(corners[:, 0], corners[:, 1], marker = '.', color="red", label="Corners", alpha = 0.4)
    for i, corner in enumerate(["BL", "BR", "TL", "TR"]):
        plt.text(corners[i, 0], corners[i, 1], corner, fontsize=12, ha='center')
    plt.legend()
    plt.show()'''
    nodedegree = np.sum(data, axis = 1)
    values, counts = np.unique(nodedegree, return_counts = True)
    #print(values, counts)
    pnodedegree = counts/np.sum(counts)
    
    entropy = -np.sum(pnodedegree*np.log(pnodedegree))
    #print(entropy)
    nonzeroindices = np.argwhere(data ==1)
    #print(nonzeroindices)

    edgeweights = (np.sqrt(np.sum(np.square(positions[nonzeroindices[:,0]] - positions[nonzeroindices[:,1]]), axis = 1)))
    
    print(nonzeroindices)
    print(edgeweights)
    
    
    df = pd.DataFrame(data, index = range(0,200), columns = range(0,200))

    graph = nx.from_pandas_adjacency(df, create_using=None)
    print(graph.edges())
    for m in range(len(edgeweights)):
        graph[int(nonzeroindices[m,0])][int(nonzeroindices[m,1])]['weight'] = edgeweights[m]
    
    BC = nx.betweenness_centrality(graph)
    DC = nx.degree_centrality(graph)
    CC = nx.closeness_centrality(graph)
    C = nx.clustering(graph)
    SPA = nx.shortest_path(graph, source=A[0], target=A[1]) ##unweighted
    SPB = nx.shortest_path(graph, source=B[0], target=B[1])
    ReffA = nx.resistance_distance(graph, nodeA=A[0], nodeB=A[1], weight = 'weight', invert_weight = True)
    ReffB = nx.resistance_distance(graph, nodeA=B[0], nodeB=B[1], weight = 'weight', invert_weight = True)
    
    #print(positions.shape[0], np.average(list(BC.values())), np.average(list(DC.values())), np.average(list(CC.values())), np.average(list(C.values())), len(SPA), len(SPB))
    
    networkmeasures[f,:] = np.asarray((len(positions), np.average(list(BC.values())), np.average(list(DC.values())), np.average(list(CC.values())), np.average(list(C.values())), len(SPA), len(SPB), np.average(nodedegree), np.sum(edgeweights), entropy, ReffA, ReffB))


#ax2 = ax.twinx()
#ax2.plot(networkmeasures[:,11], '-.')
#plt.show()
#exit()
figure, ax = plt.subplots(ncols = 2)
ax[0].plot(networkmeasures[:,5])
ax[1].plot(networkmeasures[:,6])
ax[0].set(ylabel = r'$\textrm{Shortest Path}$',xlabel = 'Lloyd', title = 'Config A')
#ax[0].subtitle(A)
ax[1].set(ylabel = r'$\textrm{Shortest Path}$',xlabel = 'Lloyd', title = 'Config B')
#ax[1].subtitle(B)
plt.show()


figure, ax = plt.subplots(nrows = 4, ncols = 4, figsize = (8,3))
figure.subplots_adjust(right = 0.99, left = 0.096, bottom = 0.07, top  =0.99, wspace = 0.729) 
ax[0,0].scatter(positions[:,0], positions[:,1], c =list(BC.values()))
ax[0,0].set(title = 'BC', ylim = (0, 2000), xlim = (0, 2000), xticklabels = [], yticklabels = []),
ax[0,0].axis('equal')
ax[0,1].scatter(positions[:,0], positions[:,1], c =list(DC.values()))
ax[0,1].set(title = 'DC',ylim = (0, 2000), xlim = (0, 2000), xticklabels = [], yticklabels = [])
ax[0,1].axis('equal')
ax[0,2].set(title = 'CC',ylim = (0, 2000), xlim = (0, 2000), xticklabels = [], yticklabels = [])
ax[0,2].axis('equal')
ax[0,2].scatter(positions[:,0], positions[:,1], c =list(CC.values()))
ax[0,3].set(title = 'C',ylim = (0, 2000), xlim = (0, 2000), xticklabels = [], yticklabels = [])
ax[0,3].axis('equal')
ax[0,3].scatter(positions[:,0], positions[:,1], c =list(C.values()))
ax[1,0].plot(networkmeasures[:,1])
ax[1,1].plot(networkmeasures[:,2])
ax[1,2].plot(networkmeasures[:,3])
ax[1,3].plot(networkmeasures[:,4])
ax[1,3].set(ylabel = r'$<C>$',xlabel = 'Lloyd')
ax[1,2].set(ylabel = r'$<CC>$',xlabel = 'Lloyd')
ax[1,1].set(ylabel = r'$<DC>$',xlabel = 'Lloyd')
ax[1,0].set(ylabel = r'$<BC>$',xlabel = 'Lloyd')
ax[2,0].plot(networkmeasures[:,5])
ax[2,1].plot(networkmeasures[:,6])
ax[2,2].plot(networkmeasures[:,7])
ax[2,3].plot(networkmeasures[:,8])
ax[2,0].set(ylabel = r'$SPA$',xlabel = 'Lloyd')
ax[2,1].set(ylabel = r'$SPB$',xlabel = 'Lloyd')
ax[2,2].set(ylabel = r'$<ND>$',xlabel = 'Lloyd')
ax[2,3].set(ylabel = r'$EW$',xlabel = 'Lloyd')
ax[3,2].plot(networkmeasures[:,9])
ax[3,1].plot(networkmeasures[:,11])
ax[3,3].plot(networkmeasures[:,0])
ax[3,0].plot(networkmeasures[:,10])
ax[3,0].set(ylabel = r'$ReffA$',xlabel = 'Lloyd')
ax[3,1].set(ylabel = r'$ReffB$',xlabel = 'Lloyd')
ax[3,2].set(ylabel = r'$Node entropy$',xlabel = 'Lloyd')
ax[3,3].set(ylabel = r'$N$',xlabel = 'Lloyd')
plt.show()

