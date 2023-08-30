#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
from scipy.stats import norm
import os


# In[31]:


rainfall_evidences = np.loadtxt("data.txt", dtype=float) 
rainfall_evidences = rainfall_evidences.reshape(-1,1)
# rainfall_evidences.shape


# In[32]:


parameters_file = "parameters.txt"
num_states = 0
with open(parameters_file) as f:
    num_states = int(f.readline().strip())
# num_states


# In[33]:


transition_matrix = np.loadtxt(parameters_file , dtype = float , skiprows = 1 , max_rows = num_states )
# transition_matrix


# In[34]:


means = np.loadtxt(parameters_file , dtype = float , skiprows = num_states + 1  , max_rows = 1 ).reshape(-1,1)
# means


# In[35]:


standard_deviations = np.loadtxt(parameters_file , dtype = float , skiprows = num_states + 2  , max_rows = 1 ).reshape(-1,1)
# standard_deviations


# In[36]:


def stationary_distribution(transition_matrix):
    #note: the matrix is row stochastic.
    #A markov chain transition will correspond to left multiplying by a row vector.
    Q = np.array(transition_matrix)

    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary_distribution = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary_distribution = stationary_distribution.real
    
    return stationary_distribution


# In[37]:


def Viterbi_Implementation(evidences, transition_matrix,initial_probaility,mu,sigma):
    
    num_states = transition_matrix.shape[0]
    num_evidences = evidences.shape[0]
    
    
    prev_state = np.full((num_states, num_evidences), -1)
    previous_prob = np.empty(num_states)
    current_prob  = np.empty(num_states)
    
    
    i = 0
    while i < num_states:
        previous_prob[i] = initial_probability[i] * norm.pdf(x=evidences[0], loc=mu[i], scale=sigma[i])
        i = i+1
    
    i=1
    while i<num_evidences:
        j=0
        while j<num_states:
            temp = transition_matrix[0, j] * norm.pdf(x=evidences[i], loc=mu[j], scale=sigma[j])
            current_prob[j] = previous_prob[0] * temp
            prev_state[j, i] = 0
            
            k=1
            while k<num_states:
                temp = transition_matrix[k, j] * norm.pdf(x=evidences[i], loc=mu[j], scale=sigma[j])
                present_state = previous_prob[k] * temp
                
                if present_state > current_prob[j]:
                    current_prob[j] = present_state
                    prev_state[j, i] = k
                k = k+1
            
            j = j+1
        previous_prob[:] = current_prob
        i = i+1
        
    probable_states = [current_prob.argmax()]
    
    i = 0
    while i < num_evidences-1:
        probable_states.append(prev_state[probable_states[i], num_evidences - (i + 1)])
        i = i+1
        
    probable_states.reverse()
    return probable_states


# In[38]:


initial_probability = stationary_distribution(transition_matrix)
# initial_probability


# In[39]:


hidden_states = Viterbi_Implementation(rainfall_evidences, transition_matrix,initial_probability, means,standard_deviations)
hidden_states = ['"El Nino"' if hidden_state == 0 else '"La Nina"' for hidden_state in hidden_states]

with open('estimated_most_likely_hidden_states.txt', 'w') as output_file:
    output_file.write('\n'.join(hidden_states))


# In[ ]:





# In[ ]:




