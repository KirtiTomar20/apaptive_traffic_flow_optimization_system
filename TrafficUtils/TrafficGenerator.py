import numpy as np
import math


class TrafficGenerator:
    def __init__(self, max_steps):
        self._n_cars_generated = 1000  # how many cars per episode
        self._max_steps = max_steps

    # generation of routes of cars
    def generate_routefile(self, seed):
        
        if seed >=0 :
            np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - min_old) + min_new)

        car_gen_steps = np.rint(car_gen_steps) # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open(r"C:\Users\user\Desktop\DQL-TSC-master\intersectionConfigs\vehicles.trips.4Lanes.xml", "w") as routes:
            print("""<routes>
            <vType id="type1" vClass="bicycle"/>
            <vType id="type2" vClass="motorcycle"/>
            <vType id="type3" vClass="passenger"/>
            <vType id="type4" vClass="truck"/>
            <vType id="type5" vClass="bus"/>
        
            <route id="r0" edges="51o 1i 2o 52i"/>
            <route id="r1" edges="51o 1i 4o 54i"/>
            <route id="r2" edges="51o 1i 3o 53i"/>
            <route id="r3" edges="54o 4i 3o 53i"/>
            <route id="r4" edges="54o 4i 1o 51i"/>
            <route id="r5" edges="54o 4i 2o 52i"/>
            <route id="r6" edges="52o 2i 1o 51i"/>
            <route id="r7" edges="52o 2i 4o 54i"/>
            <route id="r8" edges="52o 2i 3o 53i"/>
            <route id="r9" edges="53o 3i 4o 54i"/>
            <route id="r10" edges="53o 3i 1o 51i"/>
            <route id="r11" edges="53o 3i 2o 52i"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                type_random = np.random.choice(["type1", "type2", "type3", "type4", "type5"])
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination

                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="%s" route="r0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="%s" route="r6" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    # elif route_straight == 3:
                    #     print('    <vehicle id="N_S_%i" type="%s" route="r3" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="%s" route="r9" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="%s" route="r1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="%s" route="r2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, type_random,step), file=routes)
                    # elif route_turn == 3:
                    #     print('    <vehicle id="N_W_%i" type="%s" route="r4" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    # elif route_turn == 4:
                    #     print('    <vehicle id="N_E_%i" type="%s" route="r5" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="%s" route="r7" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="%s" route="r8" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="%s" route="r10" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="%s" route="r11" depart="%s" departLane="random" departSpeed="10" />' % (car_counter,type_random, step), file=routes)

            print("</routes>", file=routes)
