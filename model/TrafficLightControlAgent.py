from model.DeepQNetworkModel import DeepQNetworkModel
import traci
import tensorflow as tf
from JobUtils.SumoController import SumoSimulationManager
import numpy as np
import random
from collections import deque
import copy


# import traci.constants as tc
class TrafficLightControlAgent:

    def __init__(self, env, traffic_gen, max_steps, num_experients, total_episodes, qmodel_filename, stats,stats_filename, init_epoch,
                 learn=True):

        self.env = env
        self.traffic_gen = traffic_gen
        self.total_episodes = total_episodes
        self.discount = 0.75
        self.epsilon = 0.9
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 100
        self.num_states = 80
        self.num_actions = 9
        self.num_experiments = num_experients

        self.cycle_duration = 200
        self.stats = stats
        self.init_epoch = 0
        self.QModel = None
        self.tau = 20
        self.TargetQModel = None
        self.qmodel_filename = qmodel_filename
        self.stats_filename = stats_filename
        self.init_epoch = init_epoch
        self._load_models(learn)
        self.max_steps = max_steps

    def _load_models(self, learn=True):

        self.QModel = DeepQNetworkModel(self.num_states, self.num_actions)
        self.TargetQModel = DeepQNetworkModel(self.num_states, self.num_actions)

        if self.init_epoch != 0 or not learn:
            print('Model read from file')
            try:
                qmodel_fd = tf.keras.models.load_model(self.qmodel_filename)
                self.QModel = qmodel_fd
                self.TargetQModel = qmodel_fd
            except (OSError, ValueError) as e:
                print(f"Error loading model: {e}")

        return self.QModel, self.TargetQModel

    def _preprocess_input(self, state):
        state = np.reshape(state, [1, self.num_states])
        return state

    def _add_to_replay_buffer(self, curr_state, action, reward, next_state, done):
        self.replay_buffer.append((curr_state, action, reward, next_state, done))

    def _sync_target_model(self):
        self.TargetQModel.set_weights(self.QModel.get_weights())

    def _replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

        for i in range(len(mini_batch)):
            curr_state, action, reward, next_state, done = mini_batch[i]
            y_target = self.QModel.predict(curr_state)  # get existing Q_values for the current state
            y_target[0][action] = reward if done else reward + self.discount * np.max(self.TargetQModel.predict(
                next_state))  # modify the Q_values for the action performed to get the new target
            x_batch.append(curr_state[0])
            y_batch.append(y_target[0])
        # print(x_batch)
        # print(y_batch)
        self.QModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def _select_action(self, episode, state, learn=True):
        if learn:
            epsilon = 1 - episode / self.total_episodes
            choice = np.random.random()
            no_action = copy.deepcopy(self.num_actions)
            if choice <= epsilon:
                action = np.random.choice(range(no_action))
            else:
                action = np.argmax(self.QModel.predict(state))
        else:
            print("Model:", self.QModel.predict(state))
            # print(self.QModel.predict(state))
            action = np.argmax(self.QModel.predict(state))

        return action

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action):
        """
            Sets the traffic light phase Time based on the action and ensures durations stay within limits.

            Args:
                action: Integer representing the chosen action (0-8).
            """
        cycle_duration_1 = 205
        cycle_duration_2 = 195
        cycle_duration_std = 200
        # Handle action selection and update durations
        if action == 0:
            # Do nothing (action doesn't change any phase duration)
            print("No change in phase cycle time")
            traci.trafficlight.setProgram("0", "0")
            return cycle_duration_std
        elif action == 1:
            # Increase northbound-southbound through phase (t1) by 5 seconds, but limit to max
            print("Increased northbound-southbound Green Light Time due to more traffic")
            traci.trafficlight.setProgram("0", "1")
            return cycle_duration_1
        elif action == 2:
            print("Decreased northbound-southbound Green Light Time due to less traffic")
            # Decrease northbound-southbound through phase (t1) by 5 seconds
            traci.trafficlight.setProgram("0", "2")
            return cycle_duration_2
        elif action == 3:
            print("Increased eastbound-westbound Green Light Time due to more traffic")
            # Increase eastbound-westbound through phase (t3) by 5 seconds
            traci.trafficlight.setProgram("0", "3")
            return cycle_duration_1
        elif action == 4:
            print("Decreased eastbound-westbound Green Light Time due to less traffic")
            # Decrease eastbound-westbound through phase (t3) by 5 seconds
            traci.trafficlight.setProgram("0", "4")
            return cycle_duration_2

        elif action == 5:
            # Increase southbound left-turn/northbound left-turn phase (t2) by 5 seconds
            print("Increased southbound left-turn/northbound left-turn Green Light Time due to more traffic")
            traci.trafficlight.setProgram("0", "5")
            return cycle_duration_1
        elif action == 6:
            # Decrease southbound left-turn/northbound left-turn phase (t2) by 5 seconds
            print("Decreased southbound left-turn/northbound left-turnGreen Light Time due to less traffic")
            traci.trafficlight.setProgram("0", "6")
            return cycle_duration_2
        elif action == 7:
            # Increase westbound left-turn/eastbound left-turn phase (t4) by 5 seconds
            print("Increased westbound left-turn/eastbound left-turn Green Light Time due to more traffic")
            traci.trafficlight.setProgram("0", "7")
            return cycle_duration_1
        elif action == 8:
            # Decrease westbound left-turn/eastbound left-turn phase (t4) by 5 seconds
            print("Decreased westbound left-turn/eastbound left-turn Green Light Time due to less traffic")
            traci.trafficlight.setProgram("0", "8")
            return cycle_duration_2
        else:
            raise ValueError("Invalid action value")

    def evaluate_model(self, experiment, seeds, init_epoch):

        self.traffic_gen.generate_routefile(seeds[init_epoch])
        curr_state = self.env.start()

        for e in range(init_epoch, self.total_episodes):
            done = False
            intersection_queue_sum = 0
            negative_rewards_sum = 0
            old_action = None
            while not done:
                curr_state = copy.deepcopy(self._preprocess_input(curr_state))
                action = copy.deepcopy(self._select_action(e, curr_state, learn=False))
                duration = copy.deepcopy(self._set_green_phase(action))
                reward, next_state, done = copy.deepcopy(self.env.calculate_reward(duration))
                print("Current Reward", reward)
                next_state = copy.deepcopy(self._preprocess_input(next_state))

                curr_state = next_state
                print(curr_state)
                old_action = action
                _, _, _, _, total = copy.deepcopy(self.env.get_intersection_q_per_step())
                intersection_queue_sum += total
                if reward < 0:
                    negative_rewards_sum += reward

            # self._save_stats(experiment, e, sum_intersection_queue,sum_neg_rewards)
            print('negative_rewards_sum={}'.format(negative_rewards_sum))
            print('intersection_queue_sum={}'.format(intersection_queue_sum))
            print('Epoch {} complete'.format(e))
            if e + 1 < self.total_episodes:
                self.traffic_gen.generate_routefile(seeds[e + 1])
            curr_state = self.env.reset()

    def execute_without_model(self, experiment, seeds, init_epoch):
        self.traffic_gen.generate_routefile(seeds[self.init_epoch])
        self.env.start()

        for e in range(init_epoch, self.total_episodes):
            done = False
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            while not done:
                duration = copy.deepcopy(self._set_green_phase(0))
                reward, _, done = copy.deepcopy(self.env.calculate_reward(duration))
                if reward < 0:
                    sum_neg_rewards += reward
                _, _, _, _, total = copy.deepcopy(self.env.get_intersection_q_per_step())
                sum_intersection_queue += total

            self._save_stats(experiment, e, sum_intersection_queue, sum_neg_rewards)
            print('sum_neg_rewards={}'.format(sum_neg_rewards))
            print('sum_intersection_queue={}'.format(sum_intersection_queue))
            print('Epoch {} complete'.format(e))
            # if e != 0:
            #     os.remove('stats_classical{}_{}.npy'.format(experiment, e - 1))
            # elif experiment != 0:
            #     os.remove('stats_classical{}_{}.npy'.format(experiment - 1, self.total_episodes - 1))
            if e + 1 < self.total_episodes:
                self.traffic_gen.generate_routefile(seeds[e + 1])
            self.env.reset()

    def train(self, experiment, init_epoch):
        print("Inside Train")
        self.traffic_gen.generate_routefile(0)
        curr_state = self.env.start()
        for e in range(init_epoch, self.total_episodes):
            print("episode:", e)
            curr_state = copy.deepcopy(self._preprocess_input(curr_state))
            old_action = None
            done = False  # whether the episode has ended or not
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            round = 0
            while not done:
                if (round == 0):
                    action = 0
                else:
                    action = copy.deepcopy(self._select_action(e, curr_state))
                duration = copy.deepcopy(self._set_green_phase(action))
                reward, next_state, done = copy.deepcopy(self.env.calculate_reward(duration))
                print("Current Reward", reward)
                next_state = copy.deepcopy(self._preprocess_input(next_state))
                self._add_to_replay_buffer(curr_state, action, reward, next_state, done)
                if e > 0 and e % self.tau == 0:
                    self._sync_target_model()
                self._replay()
                curr_state = copy.deepcopy(next_state)
                old_action = action
                _, _, _, _, total = copy.deepcopy(self.env.get_intersection_q_per_step())
                sum_intersection_queue += total
                if reward < 0:
                    sum_neg_rewards += reward
                round += 1
                # Check if the simulation time reaches 5399
                if SumoSimulationManager.get_simulation_Time() == 4599:
                    done = True  # Terminate the episode if simulation time reaches 5399

            print('current reward:', sum_neg_rewards, 'current queue:', sum_intersection_queue)
            self._save_stats(experiment, e, sum_intersection_queue, sum_neg_rewards)
            self.QModel.save('qmodel_{}_{}.hd5'.format(experiment, e))

            self.traffic_gen.generate_routefile(e + 1)
            curr_state = self.env.reset()  # Reset the environment before every episode

        print('Training complete')

    def _save_stats(self, experiment, episode, sum_intersection_queue_per_episode, sum_rewards_per_episode):
        print("expt: ", experiment - 1)
        print("episodes:", episode)
        # print("stats rewards shape:", stats['rewards'].shape)
        # print(stats)
        self.stats['rewards'][experiment - 1, episode] = sum_rewards_per_episode
        self.stats['intersection_queue'][experiment - 1, episode] = sum_intersection_queue_per_episode
        np.save('stats_{}_{}.npy'.format(experiment, episode), copy.deepcopy(self.stats))
        print(self.stats)
