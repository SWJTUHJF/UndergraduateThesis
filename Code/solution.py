import os
import pickle
from datetime import datetime
from NPUE import Network, lower_problem
import random
import numpy as np


class SimulatedAnnealing:
    def __init__(self, initial_temp, cooling_rate, num_iteration, max_non_improved_times, net, block_num, multi_num,
                 initial_num, max_path_number, min_path_number, lower_problem_gap):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_iteration = num_iteration
        self.max_non_improved_times = max_non_improved_times
        self.net: Network = net
        self.block_num = block_num
        self.multi_num = multi_num
        self.initial_num = initial_num
        self.max_path_number = max_path_number
        self.min_path_number = min_path_number
        self.lower_problem_gap = lower_problem_gap

    def single_point_disruption(self, ppn):
        new_ppn = ppn.copy()
        loc = np.random.randint(low=0, high=len(ppn))
        if new_ppn[loc] == self.max_path_number:
            new_ppn[loc] -= 1
        elif new_ppn[loc] == self.min_path_number:
            new_ppn[loc] += 1
        else:
            new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    def multi_point_disruption(self, ppn):
        new_ppn = ppn.copy()
        locs = np.random.randint(low=0, high=len(ppn), size=self.multi_num)
        for loc in locs:
            if new_ppn[loc] == self.max_path_number:
                new_ppn[loc] -= 1
            elif new_ppn[loc] == self.min_path_number:
                new_ppn[loc] += 1
            else:
                new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    def block_disruption(self, ppn):
        new_ppn = ppn.copy()
        left = np.random.randint(low=0, high=len(ppn))
        right = min(len(self.net.OD), left+self.block_num)
        for loc in range(left, right):
            if new_ppn[loc] == self.max_path_number:
                new_ppn[loc] -= 1
            elif new_ppn[loc] == self.min_path_number:
                new_ppn[loc] += 1
            else:
                new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    def search(self):
        # initialize the parameters
        cur_solution = [self.initial_num] * len(self.net.OD)
        cur_temperature = self.initial_temp
        cur_objective = lower_problem(self.net, self.lower_problem_gap, ppn=cur_solution)
        best_solution = cur_solution.copy()
        best_objective = cur_objective
        obj = [best_objective]
        # log out the information
        cur_time = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
        directories = f"results\\SA{cur_time}\\solution_lists"
        os.makedirs(directories)
        with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
            file.write(f"Simulation Annealing Process\nInitial temperature={self.initial_temp}\n")
            file.write(f"Cooling rate={self.cooling_rate}\nMax iteration={self.num_iteration}\n")
            file.write(f"Network {self.net.name}\n\n")
        # main loop
        non_improve_times = 0
        for iter_times in range(self.num_iteration):
            if iter_times % 10 == 0:
                print(f'iteration {iter_times}, temperature={cur_temperature}, cur_obj={cur_objective}')
            # find a neighborhood
            if iter_times < 0.5 * self.num_iteration:  # initial stage
                candidate = self.block_disruption(cur_solution)
            elif iter_times < 0.8 * self.num_iteration:  # middle stage
                candidate = self.multi_point_disruption(cur_solution)
            else:  # later stage
                candidate = self.single_point_disruption(cur_solution)
            candidate_objective = lower_problem(self.net, self.lower_problem_gap, ppn=candidate)
            # search
            print(cur_objective - candidate_objective, cur_objective, best_objective)

            if candidate_objective < cur_objective:
                cur_solution = candidate.copy()
                cur_objective = candidate_objective
                if cur_objective < best_objective:
                    best_solution = cur_solution.copy()
                    best_objective = cur_objective
                    obj.append(best_objective)
                    print(f'**********A better solution is found: TSTT = {best_objective}**********')
                    with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
                        file.write(f"A better solution is found in iteration {iter_times}:\n")
                        file.write(f"Total system travel time is {best_objective}\n")
                        file.write(f'Corresponding solution is:\n{best_solution}\n')
                    with open(f"{directories}\\{iter_times}iter.pkl", "wb") as file:
                        pickle.dump(cur_solution, file)
                non_improve_times = 0
            elif np.random.uniform(low=0, high=1) < np.exp((cur_objective - candidate_objective) / cur_temperature):
                cur_solution = candidate.copy()
                cur_objective = candidate_objective
                non_improve_times = 0
            else:
                non_improve_times += 1

            # cooling the temperature
            cur_temperature *= self.cooling_rate

            if non_improve_times == self.max_non_improved_times:
                print(f"The best solution remains unchanged in last {self.max_non_improved_times} iterations")
                return best_solution, best_objective, obj
            if cur_temperature <= 1e-5:
                print("The temperature has dropped to 1e-5")
                return best_solution, best_objective, obj

        return best_solution, best_objective, obj
