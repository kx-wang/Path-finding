# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:14:44 2020

@author: Cathy

Implements the A* algorithm to solve for shortest path from start to finish in a maze. 
If the heuristic function h is set to zero, then Dijkstra's method is recovered

Ref: https://en.wikipedia.org/wiki/A*_search_algorithm
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


def calc_h(curNode, goal):
    # using the Manhatan Distance Heuristic, ignores any obstacles 
    x_start, y_start = curNode
    x_goal, y_goal = goal 
    h = abs(x_start-x_goal) + abs(y_start-y_goal)
    return h 

def reorder(toExamine, fScore, counter): 
    ex = toExamine[0:counter]
    
    f = np.zeros(shape = (counter))
    for i in range(counter):
        f[i] = fScore[int(ex[i,0]),int(ex[i,1])]
    
    f_ind = np.argsort(f)
    
    for i in range(counter):
        toExamine[i,:] = [ex[f_ind[i],0],ex[f_ind[i],1]]
    
    return toExamine 
#-----------------------------------------------------------------------------   
#  some different mazes, starting points, ending points 
n_problem = int(input("choose a maze (input 1, 2,or 3): "))
if n_problem == 1:
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
    goal = np.array([9,9])
    start = np.array([0,0])
elif n_problem == 2: 
    maze = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])
    goal = np.array([4,9])
    start = np.array([5,0])
else: 
    maze = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])
    goal = np.array([5,9])
    start = np.array([5,0])


#------------------------------------------------------------------------------------------------

# number of rows and columns of the maze 
n_row = 10
n_col = 10 

# will keep lists of cells that have/have not been examined yet
toExamine = np.zeros( shape = (1,2) ) # this will be priority list, based on distance to starting point or f score  
alreadyExamined = np.zeros( shape = (1,2)  )

curNode = start # current node    
toExamine[0] = start

distance = np.inf*np.ones(shape=(n_row,n_col)) # matrix to keep track of all the distances from a node to the starting cell 
distance[start[0],start[1]] = 0
fScore = np.inf*np.ones( shape = (n_row,n_col) )
fScore[start[0],start[1]] = calc_h(start,goal)

path = []
previous = np.zeros(shape = (n_row,n_col))

# while there are still nodes to examine      
while len(toExamine) > 0: 
    curNode = toExamine[0] # set the current node to the one that is at the top of the priority queue
    
    # get the valid neighbors of the current node 
    neighbors = get_neighbors(curNode) 
    
    # delete the current node from the list of nodes to examine 
    toExamine = np.delete(toExamine, np.where((toExamine[:,0] == curNode[0]) & (toExamine[:,1] == curNode[1])), axis=0)
    # add the current node to the list of node that have already been examined 
    alreadyExamined = np.insert(alreadyExamined, 0, curNode, axis=0)
    
    # iterate over all neighbors, and update the priority queue depending on which neighbors has min distance 
    row, col = curNode 
    dist_curNode = distance[int(row),int(col)]
    
    counter = 0 # this will be used to re-order the nodes in the toExamine list according to their fScore
    for i in range(len(neighbors)):  
              
        row, col = neighbors[i]
        alt_dist = dist_curNode + 1
        h = calc_h(neighbors[i], goal)
            
        # if there is an alternative distance of the neighbor node that is smaller than the known distance    
        # then update the record with the smaller distance        
        if alt_dist < distance[int(row),int(col)]: 
            distance[int(row), int(col)] = alt_dist
            fScore[int(row), int(col)] = alt_dist + h
            previous[int(row),int(col)] = cell2ind(curNode)
        
            # if the neighbor node is not in the list of nodes to examine, and it has a shorter distance
            # then it should be examined 
            if np.size(np.where( (toExamine[:,0] == neighbors[i,0]) & (toExamine[:,1]==neighbors[i,1]))) == 0:
                toExamine = np.insert(toExamine, 0, neighbors[i], axis=0)
                counter = counter + 1        

    if counter !=0:    
        toExamine = reorder(toExamine, fScore, counter) # prioritize/reorder the first counter number of nodes to examine, based on their f score

            
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
        
