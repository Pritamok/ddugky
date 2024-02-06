#!/usr/bin/env python
# coding: utf-8

# In[1]:


def bayes_theorem(p_b, p_g_given_b, p_g_given_not_b):
   # calculate P(not B)
   not_b = 1 - p_b
   # calculate P(G)
   p_g = p_g_given_b * p_b + p_g_given_not_b * not_b
   # calculate P(B|G)
   p_b_given_g = (p_g_given_b * p_b) / p_g
   return p_b_given_g


# In[2]:


#P(B)
p_b = 1/4
# P(G|B)
p_g_given_b = 1
# P(G|notB)
p_g_given_not_b = 1/3
# calculate P(B|G)
result = bayes_theorem(p_b, p_g_given_b, p_g_given_not_b)
# print result
print('P(B|G) = %.2f%%' % (result * 100))


# In[3]:


# calculate the probability of cancer patient and diagnostic test
 
# calculate P(A|B) given P(A), P(B|A), P(B|not A)
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
 # calculate P(not A)
 not_a = 1 - p_a
 # calculate P(B)
 p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
 # calculate P(A|B)
 p_a_given_b = (p_b_given_a * p_a) / p_b
 return p_a_given_b
 
# P(A)
p_a = 0.0002
# P(B|A)
p_b_given_a = 0.85
# P(B|not A)
p_b_given_not_a = 0.05
# calculate P(A|B)
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
# summarize
print('P(A|B) = %.3f%%' % (result * 100))


# In[5]:


import warnings

import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from IPython.display import Image


# In[6]:


x_blue = np.array([1,2,1,5,1.5,2.4,4.9,4.5])

y_blue = np.array([5,6.3,6.1,4,3.5,2,4.1,3])

z_blue = np.array([5,1.3,1.1,1,3.5,2,4.1,3])


x_red = np.array([5,7,7,8,5.5,6,6.1,7.7])

y_red = np.array([5,7.7,7,9,5,4,8.5,5.5])

z_red = np.array([5,6.7,7,9,1,4,6.5,5.5])


# We have to format this data to train with sklearn

red_points = np.array(zip(x_red,y_red,z_red))

blue_points = np.array(zip(x_blue,y_blue,z_blue))

points = np.concatenate([red_points,blue_points])

output = np.concatenate([np.ones(x_red.size),np.zeros(x_blue.size)])

# Now we want to classify following points

predictor = np.array([5.3,4.2,3.3])

# We are going to apply Bays classification theorem

classifier = GaussianNB()

classifier.fit(points,output)

print (classifier.predict([predictor]))

# Lets move into more real world example. Suppose we have a list of name. We want to classify this names into Male and Female categories . Our classification process is as show in below.

#Image(filename=‘classification.png’)


# In[7]:


import numpy as np

import pandas as pd

import nltk

# Define a function that parse csv file and return feature sets. We are using panda for parsing csv file.

def get_data(name, result=“gender”): 

    df = pd.read_csv(name)

    df[‘last_letter’] = df.apply (lambda row: row[0][-1],axis=1)

    df[‘last_two_letter’] = df.apply (lambda row: row[0][-2:],axis=1)

    df[‘last_is_vowel’] = df.apply (lambda row: int(row[0][-1] in “aeiouy”),axis=1)

    train = df.loc[:,[‘last_letter’,’last_two_letter’,’last_is_vowel’]]

    train_dicts = train.T.to_dict().values()

    genders = df.loc[:,[result]][result]

    return [(train_dict, gender) for train_dict,gender in zip(train_dicts,genders)]

# our names.txt is looks like,

df = pd.read_csv(“names.txt”)

print df


# In[14]:


# conditional probs

def cond_prob_dist(joint_probs):
    # P(A | B) = P( A and B ) / P(B)
    ## https://en.wikipedia.org/wiki/Conditional_probability
    
    """
    calculates the conditions prob. distribution where:
    joint_probs: is a joint prob distribution as pandas dataframe
    A = {index   of joint_probs} = {a1, a2, .. an }
    B = {columns of joint_probs} = {b1, b2, .. bn }
    
    
    returns:
    CPD = the conditional probability dist P(A|B) as a pandas dataframe
    """
    
    CPD = joint_probs.copy()

    # column sum
    col_totals = joint_probs.sum(axis=0)
    
    for col in col_totals.index:
        CPD[col] =   CPD[col] / col_totals.loc[col]
        
    # rename columns
    CPD.columns = [ f'b{i+1} = {x}' for i,x in enumerate(CPD.columns) ]
    CPD.index   = [ f'a{i+1} = {x}' for i,x in enumerate(CPD.index) ]
        
    return CPD.round(3)


# In[16]:


print( cond_prob_dist(JP) , '\n'*2)


# In[17]:


print( cond_prob_dist(JP.T).T  )


# In[18]:


import random

sample_size = 1000

num_families_at_least_one_girl = 0
num_families_two_girls = 0

for i in range(sample_size):
    first_child = random.choice(["boy", "girl"])
    second_child = random.choice(["boy", "girl"])
    if first_child == "girl" or second_child == "girl":
        num_families_at_least_one_girl += 1
        if first_child == "girl" and second_child == "girl":
            num_families_two_girls += 1

result = round(num_families_two_girls / num_families_at_least_one_girl, 2)
print(f"Out of {sample_size} families sampled, {num_families_at_least_one_girl} have at least one girl.")
print(f"Of these {num_families_two_girls} have two girls.")
print(f"This gives an experimental probability of {result} to two decimal places that,")
print("given at least one child is a girl, both children are girls.")


# In[20]:


import random

sample_size = 1000

num_families_at_least_one_girl = 0
num_families_two_girls = 0

for i in range(sample_size):
    one_child = random.choice(["boy", "girl"])
    other_child = random.choice(["boy", "girl"])
    if one_child == "girl":
        num_families_at_least_one_girl += 1
        if one_child == "girl" and other_child == "girl":
            num_families_two_girls += 1

result = round(num_families_two_girls / num_families_at_least_one_girl, 2)
print(f"Out of {sample_size} families sampled, {num_families_at_least_one_girl} have at least one girl.")
print("You know which child this is.")
print(f"Of these families, {num_families_two_girls} have two girls.")
print(f"This gives an experimental probability of {result} to two decimal places that both children are girls.")


# In[21]:


from numpy import random
random.seed(0)

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = float(ageDecade) / 100.0
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1


# In[22]:


totals


# In[23]:


purchases


# In[24]:


totalPurchases


# In[26]:


PEF = float(purchases[30]) / float(totals[30])
print('P(purchase | 30s): ' + str(PEF))


# In[27]:


PF = float(totals[30]) / 100000.0
print("P(30's): " +  str(PF))


# In[28]:


PE = float(totalPurchases) / 100000.0
print("P(Purchase):" + str(PE))


# In[29]:


print("P(30's, Purchase)" + str(float(purchases[30]) / 100000.0))


# In[30]:


print("P(30's)P(Purchase)" + str(PE * PF))


# In[31]:


print((purchases[30] / 100000.0) / PF)


# In[ ]:




