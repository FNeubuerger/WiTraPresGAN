#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib
from glob import glob
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# In[3]:


nodelabels=['timestamp', 'Stempel_innen_mitte', 'Stempel_aussen', 'Matrize_zarge_oben', 'Matrize_zarge_mitte','Matrize_zarge_unten', 'Werkstueck_boden', 'Werkstueck_zarge_unten' , 'Werkstueck_zarge_mitte', 'Werkstueck_zarge_oben']

filenames = glob("../data/FEM_Data/*.csv")
print(len(filenames))
train_df = []
for filename in filenames:
    df = pd.read_csv(filename, names=nodelabels, skiprows=1, index_col=False).drop_duplicates()#.drop(columns='timestamp')
    train_df.append(df.to_numpy())

print(len(train_df))   
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_df) 
print(train_data.shape)


# In[4]:


## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 10
parameters['batch_size'] = 128


# In[ ]:


generated_data = timegan(train_data, parameters)   
print('Finish Synthetic Data Generation')


# In[ ]:


metric_iteration = 5

discriminative_score = list()
for _ in range(metric_iteration):
    temp_disc = discriminative_score_metrics(train_data, generated_data)
    discriminative_score.append(temp_disc)

print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))


# In[ ]:


predictive_score = list()
for tt in range(metric_iteration):
    temp_pred = predictive_score_metrics(train_data, generated_data)
    predictive_score.append(temp_pred)   
    
print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))


# In[ ]:


scores = pd.DataFrame({'DiscriminativeScore' : discriminative_score, 'PredictiveScore' : predictive_score})


# In[ ]:


visualization(ori_data, generated_data, 'pca')
visualization(ori_data, generated_data, 'tsne')


