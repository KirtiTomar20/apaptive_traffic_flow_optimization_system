# -*- coding: utf-8 -*-

import sys
from model.DeepQNetworkModel import DeepQNetworkModel
import traci
import tensorflow as tf
from JobUtils.SumoController import SumoSimulationManager
import numpy as np
import random
from TrafficUtils.TrafficGenerator import TrafficGenerator
from model.TrafficLightControlAgent import TrafficLightControlAgent
from collections import deque
import JobUtils.JobConfigManager as utils
import copy


def run_traffic_light_control_agent():
    # --- TRAINING OPTIONS ---
    gui = True
    sys.path.append('C:/Program Files (x86)/Eclipse/Sumo/tools')

    # ----------------------

    # attributes of the agent

    # setting the cmd mode or the visual mode
    # "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
    if gui:
        sumoBinary = 'sumo-gui.exe'
    else:
        sumoBinary = 'sumo.exe'

    # initializations
    max_steps = 4600  # seconds = 1 h 30 min each episode
    total_episodes =100
    num_experiments = 2
    learn = False
    traffic_gen = TrafficGenerator(max_steps)
    qmodel_filename, stats_filename = utils.get_file_names()
    init_experiment, init_epoch = utils.get_init_epoch(stats_filename, total_episodes)
    print('init_experiment={} init_epoch={}'.format(init_experiment, init_epoch))
    stats = utils.get_stats(stats_filename, num_experiments, total_episodes)

    for experiment in range(init_experiment, num_experiments):
        env = SumoSimulationManager(sumoBinary, max_steps)
        # print(env)
        # Assuming `env` is your SumoEnv instance
        tl = TrafficLightControlAgent(env, traffic_gen, max_steps, num_experiments, total_episodes, qmodel_filename,
                                      stats, stats_filename, init_epoch, learn)
        # init_epoch = 0 # reset init_epoch after first experiment
        if learn:
            print("Inside learn")
            tl.train(experiment, init_epoch)
        else:
            seeds = np.load('seed.npy')
            tl.evaluate_model(experiment, seeds, init_epoch)
            # tl.execute_classical(experiment, seeds,init_epoch)

        stats = copy.deepcopy(tl.stats)
        print(stats['rewards'][0:experiment + 1, :])
        print(stats['intersection_queue'][0:experiment + 1, :])
        utils.plot_rewards(stats['rewards'][0:experiment + 1, :])
        utils.plot_intersection_queue_size(stats['intersection_queue'][0:experiment + 1, :])
        del env
        del tl
        print('Experiment {} complete.........'.format(experiment))


if __name__ == "__main__":
    run_traffic_light_control_agent()
