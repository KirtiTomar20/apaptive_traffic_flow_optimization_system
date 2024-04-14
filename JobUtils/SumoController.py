# -*- coding: utf-8 -*-
import traci
import numpy as np


class SumoSimulationManager:

    def __init__(self, sumoBinary=None, max_steps=None):
        self.port = 8814
        self.sumoCmd = [sumoBinary, "-c",
                        "C:/Users/user/Desktop/DQL-TSC-master/intersectionConfigs/trafficLight.4Lanes.sumocfg",
                        "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--log", "logfile.txt"]

        self.SUMO_INT_LANE_LENGTH = 100
        self.num_states = 80  # 0-79 see _encode_env_state function for details
        self.max_steps = max_steps
        self._init()

    def _init(self):
        self.current_state = None
        self.curr_WQ = 0
        self.steps = 0

    def get_state(self):
        return self.current_state

    def start(self):
        traci.start(self.sumoCmd, label="main", port=self.port)
        traci.setOrder(1)
        self.current_state = self._encode_env_state()
        return self.current_state

    def reset(self):
        print("Inside reset")
        traci.close()  # Close the TraCI connection
        print("Handler here")
        self._init()  # Re-initialize the environment
        self.start()  # Re-start the simulation and TraCI clients
        self.current_state = self._encode_env_state()  # Encode the current state
        return self.current_state

    @staticmethod
    def get_simulation_Time():
        return traci.simulation.getTime()

    def calculate_reward(self, num_steps=1):
        step_count = 0
        if self.steps + num_steps > self.max_steps:
            num_steps = self.max_steps - self.steps

        for i in range(num_steps):
            step_count += 1
            traci.simulationStep()

        print("Cycle Time:", step_count)
        self.steps += num_steps
        self.current_state = self._encode_env_state()
        # print("Inside step")
        new_q_N, new_q_S, new_q_E, new_q_W, _ = self.get_intersection_q_per_step()
        wait_time_N, wait_time_S, wait_time_E, wait_time_W, _ = self._get_waiting_time()

        # calculation of reward parameter
        new_WQ = new_q_N * wait_time_N + new_q_S * wait_time_S + new_q_W * wait_time_W + new_q_E * wait_time_E

        # calculate reward of action taken (change in cumulative waiting time between actions)

        reward = 0.9 * self.curr_WQ - new_WQ
        # normalize reward
        min_reward = -1000  # Example: minimum possible reward
        max_reward = 1000  # Example: maximum possible reward
        normalized_reward = (reward - min_reward) / (max_reward - min_reward)

        # Update the reward with the normalized value
        reward = normalized_reward
        # print("reward={}".format(reward))
        self.curr_WQ = new_WQ

        # one episode ends when all vehicles have arrived at their destination
        if self.steps < self.max_steps:
            is_terminal = False
        else:
            is_terminal = True

        return (reward, self.current_state, is_terminal)

    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_time(self):
        incoming_lanes_E = ["2i_0", "2i_1", "2i_2"]
        incoming_lanes_N = ["4i_0", "4i_1", "4i_2"]
        incoming_lanes_W = ["1i_0", "1i_1", "1i_2"]
        incoming_lanes_S = ["3i_0", "3i_1", "3i_2"]

        waiting_time_N = 0
        waiting_time_E = 0
        waiting_time_W = 0
        waiting_time_S = 0
        for lane in incoming_lanes_E:
            waiting_time_E += traci.lane.getWaitingTime(lane)
        for lane in incoming_lanes_W:
            waiting_time_W += traci.lane.getWaitingTime(lane)
        for lane in incoming_lanes_S:
            waiting_time_S += traci.lane.getWaitingTime(lane)
        for lane in incoming_lanes_N:
            waiting_time_N += traci.lane.getWaitingTime(lane)

        total_waiting_time = waiting_time_N + waiting_time_E + waiting_time_W + waiting_time_S
        print("Waiting Times-", waiting_time_N, waiting_time_S, waiting_time_E, waiting_time_W)
        return waiting_time_N, waiting_time_S, waiting_time_E, waiting_time_W, total_waiting_time

    def get_intersection_q_per_step(self):
        halt_N = traci.edge.getLastStepHaltingNumber("4i")
        halt_S = traci.edge.getLastStepHaltingNumber("3i")
        halt_E = traci.edge.getLastStepHaltingNumber("2i")
        halt_W = traci.edge.getLastStepHaltingNumber("1i")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        print("Queue Length-", halt_N, halt_S, halt_E, halt_W, intersection_queue)
        return halt_N, halt_S, halt_E, halt_W, intersection_queue

    def _encode_env_state(self):
        state = np.zeros(self.num_states)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            # print("vehicle lane pos:",lane_pos)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = self.SUMO_INT_LANE_LENGTH - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            # print("new vehicle lane pos:", lane_pos)
            lane_group = -1  # just dummy initialization
            is_car_valid = False  # flag for not detecting cars crossing the intersectionConfigs or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 10:
                lane_cell = 0
            elif lane_pos < 20:
                lane_cell = 1
            elif lane_pos < 30:
                lane_cell = 2
            elif lane_pos < 40:
                lane_cell = 3
            elif lane_pos < 50:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 70:
                lane_cell = 6
            elif lane_pos < 80:
                lane_cell = 7
            elif lane_pos < 90:
                lane_cell = 8
            elif lane_pos <= 100:
                lane_cell = 9

            # Isolate the  "turn left only" from "straight" and "right" turning lanes.
            # This is because TL lights are turned on separately for these sets
            if lane_id == "1i_0" or lane_id == "1i_1":
                lane_group = 0
            elif lane_id == "1i_2":
                lane_group = 1
            elif lane_id == "4i_0" or lane_id == "4i_1":
                lane_group = 2
            elif lane_id == "4i_2":
                lane_group = 3
            elif lane_id == "2i_0" or lane_id == "2i_1":
                lane_group = 4
            elif lane_id == "2i_2":
                lane_group = 5
            elif lane_id == "3i_0" or lane_id == "3i_1":
                lane_group = 6
            elif lane_id == "3i_2":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) +
                                   str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                is_car_valid = True
            elif lane_group == 0:
                veh_position = lane_cell
                is_car_valid = True

            if is_car_valid:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state

    def add_vehicle_to_N2TL(self, traci_client, vehicle_id, type_id="type3", departSpeed="max", departLane="best"):
        """
        Dynamically adds a vehicle to the N2TL edge with a randomly selected route among "NW", "NE", "NS".
        :param traci_client: New traci client.
        :param vehicle_id: Unique identifier for the vehicle.
        :param type_id: The vehicle type ID. Defaults to "standard_car".
        :param departSpeed: The departure speed of the vehicle. Defaults to "max" for the maximum speed.
        :param departLane: The lane on which the vehicle should start. Defaults to "best" to let SUMO decide.
        """

        # print(vehicle_id, " inside function")
        if traci_client is None:
            raise RuntimeError("Second TraCI client is not initialized.")
        # Randomly select a route ID from the three possible values
        route_ids = ["r3", "r4", "r5"]
        route_id = np.random.choice(route_ids)
        print(vehicle_id, " inside function")
        # Add the vehicle with the specified parameters

        traci_client.vehicle.add(vehID=vehicle_id, typeID=type_id, routeID=route_id, depart="now",
                                 departSpeed=10, departLane=departLane)
        print(vehicle_id, " vehicle added")

    def __del__(self):
        traci.close()
