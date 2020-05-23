# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:17:44 2020

@author: Cathy

Solving a maze using deep Q-learning coupled with experience replay 
Ref: Human-level control through deep reinforcement
learning, doi:10.1038/nature14236
"""

# import packages 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 

# -------------------------functions--------------------------------------------------------------------------
# turns the cell [row,col] into one number 
def cell2ind(cell): # done 
    a = cell[0]
    b = cell[1]
    ind = b + a*n_row
    return ind

def observe(action,state): # done 
    
    # get the new state resulting from the action 
    if action == 0: # move left 
        new_state = state + [0,-1]
    elif action == 1: # move right 
        new_state = state + [0,1]
    elif action == 2: # move up 
        new_state = state + [-1,0]
    else: # move down 
        new_state = state + [1,0]
            
    # determine the reward, depending on the state 
    if new_state[0] == goal[0] and new_state[1] == goal[1]: # reached the goal
        reward = 1
    elif new_state[0] < 0 or new_state[1] < 0 or new_state[0] > n_row-1 or new_state[1] > n_col-1: # hit boundary 
        reward = -0.8
    else: # is a valid move 
        reward = -0.04 
        
    return reward, new_state  

def random_action() : # done 
    x = np.random.choice([0,1,2,3])    
    return x

def get_batch(D, batch_size) :
    N = len(D)
    B = np.zeros(shape = (batch_size,4))
    for i in range(batch_size):
        rand_int = int( np.random.randint(N)  )  
        B[i,:] = D[rand_int,:]    
    return B 





#----------------------main script------------------------------    
# 0 = free cell, 1 = occupied cell 
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
n_row = 10
n_col = 10 

goal = np.array([9,9])
start = np.array([0,0])

gamma = 0.99 # discount factor 
eps = 1      # initial probability that the action will no follow that suggested by Q
M = 100      # maximum number of episodes 
T = 100      # maximum number of time steps 
N = n_row*n_col # the replay memory D has the capacity to store N experiences 
C = 10          # at every C steps, learn the Q model/ neural network again  
batch_size = 30 # size of sample from memory 

# D is a matrix to store the agent's experiences at each time step
# the experience has values s_t,a_t,r_t,s_{t+1} 
D = np.empty( shape = (N,4) )
D[:] = np.NaN 


# defining the DQN model: 
# this predicts the Q value for each action, given the input state  
model = keras.Sequential([
    keras.layers.Dense(10*10), # input layer, which has one unit for each state  
    keras.layers.Dense(10*10, activation='relu'),
    keras.layers.Dense(4) # output layer has 4 outputs, one for each action 
])
model.compile(optimizer='adam', loss='mse') # loss is the mean squared loss 


    
# --------- when there is not yet enough data to have a Q_pred -----------------
init_Q_data = np.zeros( shape = (M) ) 

counter = 0
  
# starting state  
state = start
   
for loops in range(300):    
    
    # select a random action 
    action = random_action() 
           
    # execute action to get reward and observe the next state 
    reward, new_state = observe(action,state) 
    
    # if game over, then start a new episode (effectively)
    if reward == 1 or reward == -0.8:
        # reset starting state  
        state = np.array( [0,0] )
    
    # store this transition info (S,A,R,S') into memory if it's not already stored 
    transition = [cell2ind(state), action, reward, cell2ind(new_state)]
    if np.size( np.where((D[:,0]==transition[0]) & (D[:,1]==transition[1]) & (D[:,2]==transition[2]) & (D[:,3]==transition[3])) )==0:
        D[counter % N, :] = transition
        init_Q_data[counter] = reward
        counter = counter + 1 # counts the number of unique transitions
    
    if counter == N:
        break 
    
    # updates for the next time step 
    state = new_state 

model.fit(D[:,0], init_Q_data, epochs = 8, batch_size = 16, verbose =0 ) # the Q_pred      
  



# --------------------------------training-------------------------------------- 
for episode in range(M): 
    
      
    state = start # reset starting state
    path = [] # this is to record the path taken 
       
    for t in range(T):    
        
        # select an action according to epsilon-greedy policy derived from Q
        if np.random.random() < eps:
            action = random_action() # with probability eps select a random action a_t
        else: 
            action = np.argmax(  model.predict( np.array([cell2ind(state)]))  ) 
        
        if eps > 0.1:
            eps = eps - 0.05
            
        # execute action to get reward and observe the next state 
        reward, new_state = observe(action,state) 
        if reward == 1 or reward == -0.8:
            break
        
        # store this transition info (S,A,R,S') into memory if it's not already stored 
        transition = [cell2ind(state), action, reward, cell2ind(new_state)]
        if np.size( np.where((D[:,0]==transition[0]) & (D[:,1]==transition[1]) & (D[:,2]==transition[2]) & (D[:,3]==transition[3])) )==0:
            D[counter % N, :] = transition
            counter = counter + 1 # counts the number of unique transitions
        
        # sample a batch of transitions from memory, and the batch size is the 
        # minimum between the batch_size set and the number of unique transitions 
        B = get_batch(D,batch_size)
        
        # update Q_target (y_j) with the batch data 
        B_states = B[:,0] # all the input states of the batch 
        
        Q_target = np.zeros( shape=(len(B_states)) )   
        for j in range(len(B)):
            reward_j = B[j,2]
            if reward_j == 1 or reward == -0.8:
                Q_target[j] = reward_j
            else: 
                Q_target[j] = reward_j + gamma * np.max( model.predict( np.array([B_states[j]]) ))
        
        # Train/update neural network model, with new weights, after every C steps 
        if C == 10: 
            Q_pred = model.fit(B_states, Q_target, epochs = 8, batch_size = 16, verbose =0 )
            loss = model.evaluate(B_states, Q_target, verbose = 0) 
            C == 1
        else: 
            C = C + 1
            
        # updates for the next time step 
        path.append(state)
        state = new_state 
        
        
        
# --------------------------------testing-------------------------------- 
      
        
        
        
        
        
        
        
        
        
        
        
        