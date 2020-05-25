# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:14:44 2020

@author: Cathy
"""

import numpy as np
import matplotlib.pyplot as plt 

def cell2ind(cell): 
    a = cell[0]
    b = cell[1]
    ind = b + a*n_row
    return ind

def get_neighbors(node):
    around = np.array([[0,-1], [0,1], [-1,0], [1,0]])
    neighbors = []
    
    for i in range(4):
        temp = node + around[i]
        
        # if not out of bounds and not a wall block
        row = int(temp[0])
        col = int(temp[1])
        if temp[0] > -1 and temp[1] > -1 and temp[0] < (n_row) and temp[1] < (n_col) and maze[row,col]!=1:
            neighbors.append(temp)   
                
    return np.array(neighbors)

def get_distance(node, neighbors, distance):
    row, col = node
    prev_dist = distance[row,col] 
    
    min_dist = 1e3
    for i in range(len(neighbors)):
        row,col = neighbors[i]
        distance[row,col] = prev_dist + 1
        if distance[row,col] < min_dist:
            min_dist = distance[row,col]
            ind_min_dist = i
        
    return distance, ind_min_dist 



#-----------------------------------------------------------------------------    

maze = np.array([
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 1,  1,  0,  1,  1,  0,  1,  0,  0,  0],
    [ 0,  0,  1,  0,  1,  0,  1,  1,  1,  0],
    [ 0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
    [ 0,  1,  1,  1,  1,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0]
])
'''
maze = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
])
goal = np.array([5,9])
start = np.array([5,0])
'''
n_row = 10
n_col = 10 
goal = np.array([9,9])
start = np.array([0,0])


# will keep lists of cells that have/have not been examined yet
toExamine = np.zeros( shape = (1,2) ) # priority list 
alreadyExamined = np.zeros( shape = (1,2)  )

curNode = start # current node    

distance = np.inf*np.ones(shape=(n_row,n_col)) # matrix to keep track of all the distances from a node to the starting cell 
distance[0,0] = 0

path = []
previous = np.zeros(shape = (n_row,n_col))

# while there are still nodes to examine      
while len(toExamine) > 0: 
    curNode = toExamine[0] # set the current node to the one that is at the top of the priority queue
    #path.append(curNode)
    #print(curNode)
    
    # get the valid neighbors of the current node 
    neighbors = get_neighbors(curNode) 
    
    # delete the current node from the list of nodes to examine 
    toExamine = np.delete(toExamine, np.where((toExamine[:,0] == curNode[0]) & (toExamine[:,1] == curNode[1])), axis=0)
    # add the current node to the list of node that have already been examined 
    alreadyExamined = np.insert(alreadyExamined, 0, curNode, axis=0)
    
    # iterate over all neighbors, and update the priority queue depending on which neighbors has min distance 
    row, col = curNode 
    dist_curNode = distance[int(row),int(col)]
    
    for i in range(len(neighbors)):  
        
        # check if the neighbor has already been examined (ie has been in priority queue and then deleted)
        # if not, then proceed with the following 
        if np.size(np.where( (alreadyExamined[:,0] == neighbors[i,0]) & (alreadyExamined[:,1]==neighbors[i,1]))) == 0:
          
            row, col = neighbors[i]
            alt_dist = dist_curNode + 1
        
            # if there is an alternative distance of the neighbor node that is smaller than the known distance    
            # then update the record with the smaller distance        
            if alt_dist < distance[int(row),int(col)]: 
                distance[int(row), int(col)] = alt_dist
                previous[int(row),int(col)] = cell2ind(curNode)
                
            # since for all the nodes, the lengths to its neighbors are the same, then for the next node 
            # to examine, the priorities will be the same 
            toExamine = np.insert(toExamine, 0, neighbors[i], axis=0)
            
    #if len(neighbors) == 0: 
    #    path.remove(path[-1])
        
    if (curNode[0] == goal[0]) and (curNode[1] == goal[1]): 
        break 


# --------------------------construct the path--------------------------------- 
ind2cell = np.zeros(shape = (n_row*n_col,2))
counter = 0
for i in range(n_col): 
    for j in range(n_col):
        ind2cell[counter,:] = [i,j]
        counter = counter + 1
        
curNode = goal 
path.append(curNode)
keepGoing = True 
while (keepGoing): 
    row, col = curNode 
    temp = previous[int(row),int(col)]
    row, col = ind2cell[int(temp),:]
    path.append([int(row),int(col)])      
    curNode = [int(row),int(col)]
    
    if (curNode[0] == start[0] and curNode[1]==start[1]):
        keepGoing = False 
    
#-----------------visualizing the solution-------------------------------------        
path_matrix = np.zeros(shape=(n_row, n_col))
for i in range(len(path)):
    k,j = path[i]
    path_matrix[int(k)][int(j)]=2
    
path_matrix = path_matrix+maze 
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(path_matrix)
ax.set_aspect('equal')
plt.show() 
        
