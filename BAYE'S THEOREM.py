#!/usr/bin/env python
# coding: utf-8

# In[1]:


def bayesTheorem(PA, PB, PBA):
    return PA * PBA / PB


# In[3]:


#define function for Bayes' theorem
def bayesTheorem(PA, PB, PBA):
    return PA * PBA / PB

#define probabilities
PRain = 0.2
PCloudy = 0.4
PCloudyRain = 0.85

#use function to calculate conditional probability
bayesTheorem(PRain, PCloudy, PCloudyRain)



# In[4]:


import pandas as pd
import numpy as np

#create pandas DataFrame with raw data
df = pd.DataFrame({'gender': np.repeat(np.array(['Male', 'Female']), 150),
                   'sport': np.repeat(np.array(['Baseball', 'Basketball', 'Football',
                                                'Soccer', 'Baseball', 'Basketball',
                                                'Football', 'Soccer']), 
                                    (34, 40, 58, 18, 34, 52, 20, 44))})

#produce contingency table to summarize raw data
survey_data = pd.crosstab(index=df['gender'], columns=df['sport'], margins=True)

#view contingency table
survey_data


# In[5]:


#extract value in second row and first column 
survey_data.iloc[1, 0]



# In[6]:


#calculate probability of being male, given that individual prefers baseball
survey_data.iloc[1, 0] / survey_data.iloc[2, 0]


# In[7]:


#calculate probability of preferring basketball, given that individual is female
survey_data.iloc[0, 1] / survey_data.iloc[0, 4]






