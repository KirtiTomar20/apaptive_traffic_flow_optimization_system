# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:55:20 2020

@author: XZ01M2
"""

import matplotlib.pyplot as plt
import numpy as np 
import glob

import pandas as pd
import seaborn as sns

def get_file_names():
     qmodel_file_name = "qmodel_1_50.hd5"
     stats_file_name = "stats_1_50.npy"

     return qmodel_file_name, stats_file_name

def get_init_epoch( filename,total_episodes ):
    
    if filename:
        index = filename.find('_')
        exp_start = index + 1 
        exp_end  = int(filename.find('_', exp_start))
        exp = int(filename[exp_start:exp_end])
        epoch_start= exp_end + 1
        epoch_end = int(filename.find('.', epoch_start))
        epoch = int(filename[epoch_start:epoch_end])
        if epoch < total_episodes -1:
            epoch +=1
        else:
            epoch = 0
            exp +=1
        
    else:
        exp=0
        epoch = 0
    return exp , epoch

def get_stats(stats_filename, num_experiments, total_episodes, learn = True):
    
    if stats_filename and learn:
        stats =np.load(stats_filename, allow_pickle = True)[()]
      
    else:
        reward_store = np.zeros((num_experiments,total_episodes))
        intersection_queue_store = np.zeros((num_experiments,total_episodes))
        stats = {'rewards': reward_store, 'intersection_queue': intersection_queue_store }

    return stats
    
    
    
def plot_sample(sample, title, xlabel, legend_label, show= True):
    
   #plt.hist(sample, bins = 5, histtype = 'bar')
    #plt.xlabel(xlabel)
    ax= sns.distplot(sample, kde=True, label =  legend_label)
    ax.set(xlabel=xlabel, title= title)
    ax.legend()
    if show:
        plt.show()
    
    
def plot_rewards( reward_store,reward_classical):
    # x = np.mean(reward_store, axis = 0 )
    plt.plot( reward_store , label = "Cummulative negative wait times-with Model",marker='o')
    plt.plot(reward_classical, label="Cummulative negative wait times-without Model")
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative negative wait times') 
    plt.title('Cummulative negative wait times across episodes') 
    plt.legend() 
    plt.show() 
    
def plot_intersection_queue_size( intersection_queue_store,interaction_queue_classical):
    
    # x = np.mean(intersection_queue_store, axis = 0 )
    plt.plot(intersection_queue_store, label = "Cummulative intersectionConfigs queue size-with Model ", color='m',marker='o')
    plt.plot(interaction_queue_classical, label="Cummulative intersectionConfigs queue size- Without Model ")
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative intersectionConfigs queue size')
    plt.title('Cummulative intersectionConfigs queue size across episodes')
    plt.legend() 
    plt.show()

