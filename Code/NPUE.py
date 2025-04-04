import heapq
import os
import pickle
import random
import re
import time
import numpy as np
from datetime import datetime
from itertools import product
from math import inf


min_path_number = 1
max_path_number = 5
line_search_gap = 1e-4
Frank_Wolfe_gap = 1e-4
lower_problem_gap = 1e-7

initial_temp = 300
initial_cooling_rate = 0.9995
later_cooling_rate = 0.995
markov_chain_length = 30
num_iteration = markov_chain_length * 3000
block_num = 20
multi_num = 5
initial_num = 3
min_temperature = 1e-2
max_non_improved_times = 100000


class Link:
    def __init__(self, link_id, tail=None, head=None, capacity=None, length=None, fft=None, b=None, power=None):
        self.link_id: int = link_id
        self.tail: Node = tail
        self.head: Node = head
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = fft
        self.b: float = b
        self.power: float = power
        self.flow: float = 0
        self.aux_flow: float = 0
        self.last_flow: float = 0
        self.cost: float = self.fft

    def __repr__(self):
        return f'Link {self.link_id} cost={self.cost} flow={self.flow}'

    def __lt__(self, other):
        if isinstance(other, Link):
            return self.link_id < other.link_id

    def update_ue_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)

    def obtain_ue_cost(self, param):
        return self.fft * (1 + self.b * (param / self.capacity) ** self.power)

    def update_so_cost(self):
        self.cost = (self.fft * (1 + self.b * (self.flow / self.capacity)) ** self.power
                     + self.fft * self.b * self.power * (self.flow / self.capacity) ** self.power)

    def obtain_so_cost(self, param):
        return (self.fft * (1 + self.b * (param / self.capacity)) ** self.power
                + self.fft * self.b * self.power * (param / self.capacity) ** self.power)

    def derivative(self):
        return self.fft * self.b * self.power * self.flow ** (self.power - 1) / self.capacity ** self.power


class Node:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.link_in: list[Link] = list()
        self.link_out: list[Link] = list()
        self.parent: Node = self
        self.dist: float = 0

    def __repr__(self):
        return f'Node {self.node_id}'


class ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: Node = origin
        self.destination: Node = destination
        self.demand: float = demand
        self.total_path_set: list[Path] = list()
        self.potential_path_set: list[Path] = list()
        self.working_path_set: list[Path] = list()

    def __repr__(self):
        return f"{self.origin.node_id}-{self.destination.node_id}:{self.demand}"

    def find_shortest_path_in_PPS(self):
        min_path = min(self.potential_path_set, key=lambda p: p.path_cost)
        return min_path.path_cost, min_path


class Path:
    def __init__(self, path_id, origin, destination, included_links):
        self.path_id: int = path_id
        self.origin: Node = origin
        self.destination: Node = destination
        self.included_links: list[Link] = included_links
        self.path_flow: float = 0
        self.path_cost: float = 0
        self.update_path_cost()

    def __repr__(self):
        nodes = [link.tail.node_id for link in self.included_links] + [self.destination.node_id]
        return f"path {self.path_id}: {'-'.join(map(str, nodes))}(cost={self.path_cost:.2f}, flow={self.path_flow})"

    def update_path_cost(self):
        self.path_cost = sum([link.cost for link in self.included_links])


class Network:
    def __init__(self, name, sst, model='UE', from_file=False):
        self.name: str = name
        self.demand_sensitivity: float = sst
        self.model = model
        self.from_file = from_file
        self.LINKS: list[Link] = list()
        self.NODES: list[Node] = list()
        self.OD: list[ODPair] = list()
        self.num_link: int = 0
        self.num_node: int = 0
        self.total_flow: float = 0
        self.read_net()
        self.read_OD()
        if self.from_file:
            self.read_flow()

    def read_flow(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_flow.txt") as file:
            lines = file.readlines()
            pattern = re.compile(r'[0-9.]+')
            data = list()
            for line in lines:
                line = pattern.findall(line)
                if len(line) != 0:
                    data.append([float(line[-2]), float(line[-1])])
            for index, value in enumerate(data):
                self.LINKS[index+1].flow = value[0]
                self.LINKS[index+1].cost = value[1]


    def generate_total_path_set(self):
        for rs in self.OD:
            k_paths = Yen(self, rs.origin.node_id, rs.destination.node_id)
            rs.total_path_set = [Path(i+1, rs.origin, rs.destination, k_paths[i]) for i in range(max_path_number)]

    def read_net(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_net.txt") as f:
            # read network information
            lines = f.readlines()
            pattern = re.compile(r"[0-9A-Za-z.~]+")
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i in range(len(lines)):
                line = lines[i]
                if "NUMBER" and "NODES" in line:
                    self.num_node = int(line[-1])
                if "NUMBER" and "LINKS" in line:
                    self.num_link = int(line[-1])
                if "~" and "capacity" in line:
                    lines = lines[i + 1:]
                    break
        # create NODE and LINK instance
        self.NODES = [Node(i) for i in range(self.num_node + 1)]
        self.LINKS = [Link(0)]
        for i, line in enumerate(lines):
            tail, head = self.NODES[int(line[0])], self.NODES[int(line[1])]
            link = Link(i+1, tail, head, float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]))
            self.LINKS.append(link)
            tail.link_out.append(link)
            head.link_in.append(link)

    def read_OD(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_trips.txt") as f:
            # read OD pairs
            lines = f.readlines()
            pattern = re.compile(r'[a-zA-Z0-9.]+')
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i, line in enumerate(lines):
                if "TOTAL" in line:
                    total_flow = float(line[-1])
                if "Origin" in line:
                    lines = lines[i:]
                    break
            for line in lines:
                if "Origin" in line:
                    origin = self.NODES[int(line[-1])]
                else:
                    for i in range(len(line) // 2):
                        destination = self.NODES[int(line[2 * i])]
                        demand = float(line[2 * i + 1])
                        if demand != .0:
                            self.OD.append(ODPair(origin, destination, demand * self.demand_sensitivity))
            if total_flow * self.demand_sensitivity != sum([od.demand for od in self.OD]):
                raise ValueError("Demand data is wrong")
            else:
                self.total_flow = total_flow

    def update_all_link_cost(self):
        for link in self.LINKS[1:]:
            if self.model == "UE":
                link.update_ue_cost()
            elif self.model == "SO":
                link.update_so_cost()
            else:
                raise ValueError("Choose the right model!")

    def update_all_path_cost(self):
        for od in self.OD:
            for path in od.potential_path_set:
                path.update_path_cost()


class FW:
    def __init__(self, net, model):
        self.net: Network = net
        self.model = model
        self.gap_list = list()
        self.run()
        self.ttst = sum([link.flow * link.obtain_ue_cost(link.flow) for link in self.net.LINKS[1:]])

    def run(self):
        iter_times = 0
        self.initialize()
        gap = inf
        while iter_times == 0 or gap >= Frank_Wolfe_gap:
            self.update_cost()
            self.all_or_nothing()
            step = self.line_search()
            self.update_flow(step)
            iter_times += 1
            gap = self.converge()
            self.gap_list.append(gap)

    def initialize(self):
        for link in self.net.LINKS[1:]:
            link.flow, link.aux_flow = 0, 0
        self.update_cost()
        self.all_or_nothing()
        for link in self.net.LINKS[1:]:
            link.flow = link.aux_flow
            link.aux_flow = 0

    def all_or_nothing(self):
        for od in self.net.OD:
            tail, head, demand = od.origin, od.destination, od.demand
            shortest_path = dijkstra(tail.node_id, head.node_id, self.net)
            for link in shortest_path:
                link.aux_flow += demand

    def update_cost(self):
        for link in self.net.LINKS[1:]:
            if self.model == "UE":
                link.update_ue_cost()
            elif self.model == "SO":
                link.update_so_cost()
            else:
                raise ValueError()

    def line_search(self):
        def derivative(step):
            res = 0
            for link in self.net.LINKS[1:]:
                if self.model == "UE":
                    res += (link.aux_flow - link.flow) * link.obtain_ue_cost(
                        link.flow + step * (link.aux_flow - link.flow))
                elif self.model == "SO":
                    res += (link.aux_flow - link.flow) * link.obtain_so_cost(
                        link.flow + step * (link.aux_flow - link.flow))
            return res
        left, right, mid = 0, 1, 0.5
        while abs(derivative(mid)) >= line_search_gap:
            if derivative(mid) * derivative(right) > 0:
                right = mid
            elif derivative(mid) * derivative(left) > 0:
                left = mid
            else:
                raise ValueError("Something is wrong with line search!")
            mid = (right + left) / 2
        return mid

    def update_flow(self, step):
        for link in self.net.LINKS[1:]:
            link.last_flow = link.flow
            link.flow = link.flow + step * (link.aux_flow - link.flow)
            link.aux_flow = 0

    def converge(self):
        SPTT = 0
        for od in self.net.OD:
            shortest_path = dijkstra(od.origin.node_id, od.destination.node_id, self.net)
            SPTT += sum([link.cost for link in shortest_path]) * od.demand
        TSTT = sum([link.flow * link.cost for link in self.net.LINKS[1:]])
        cur_gap = (TSTT / SPTT) - 1
        print(cur_gap)
        return cur_gap


def dijkstra(o_id: int, d_id: int, net: Network, forbidden_nodes: list[Node] = None, forbidden_link: Link = None):
    # initialize
    for node in net.NODES:
        node.parent = node
        node.dist = inf
    net.NODES[o_id].parent = -1
    net.NODES[o_id].dist = 0
    # main loop
    SEL = [net.NODES[o_id]]
    while SEL:
        SEL.sort(key=lambda n: n.dist, reverse=True)
        cur = SEL.pop()
        if cur.node_id == d_id:
            break
        for link in cur.link_out:
            next_node = link.head
            if forbidden_nodes:
                if next_node in forbidden_nodes:
                    continue
            if forbidden_link:
                if link == forbidden_link:
                    continue
            if link.head.dist > cur.dist + link.cost:
                link.head.parent = cur
                link.head.dist = cur.dist + link.cost
                if link.head not in SEL:
                    SEL.append(link.head)
    # obtain the shortest path
    shortest_path = []
    cur = net.NODES[d_id]
    while cur.parent != -1:
        p = cur.parent
        for link in cur.link_in:
            if link.tail == p:
                temp = link
                break
        else:
            return None
        shortest_path.append(temp)
        cur = p
    shortest_path.reverse()
    return shortest_path


def Yen(net: Network, o_id: int, d_id: int):
    A = []  # to store the k-paths
    path = dijkstra(o_id, d_id, net)
    A.append(path)
    B = []  # to store the obtained possible paths
    existing_paths = [path]
    while len(A) != max_path_number:
        previous_path = A[-1]
        for spur_link_index in range(len(previous_path)):
            spur_node = previous_path[spur_link_index].tail.node_id  # deviate for each link's tail node
            root_path = previous_path[:spur_link_index]
            # calculate the cost of the root path
            root_cost = sum([link.cost for link in root_path])
            # remove the adjacent link and root nodes
            forbidden_nodes = [link.tail for link in root_path]
            forbidden_link = previous_path[spur_link_index]
            # find the spur path
            spur_path = dijkstra(spur_node, d_id, net, forbidden_nodes, forbidden_link)
            if spur_path is None:
                continue
            total_path = root_path + spur_path
            if total_path in existing_paths:
                continue
            else:
                total_cost = root_cost + sum([link.cost for link in spur_path])
                existing_paths.append(total_path)
                heapq.heappush(B, (total_cost, total_path))
        if not B:
            raise ValueError("There's no k paths in the network")
        cost, path = heapq.heappop(B)
        A.append(path)
    return A


def lower_problem(net: Network, ppn):
    if isinstance(ppn[0], int):
        for i, od in enumerate(net.OD):
            od.potential_path_set = od.total_path_set[:ppn[i]]
    else:
        for i, od in enumerate(net.OD):
            od.potential_path_set = [path for j, path in enumerate(od.total_path_set) if ppn[i][j] == 1]
    for link in net.LINKS[1:]:
        link.flow, link.last_flow = 0, 0
        link.update_ue_cost()
    # Step 1: Initialization
    for od in net.OD:
        od.working_path_set = list()
        for path in od.potential_path_set:
            path.update_path_cost()
            path.path_flow = 0
        min_dist, min_path = od.find_shortest_path_in_PPS()
        od.working_path_set.append(min_path)
        min_path.path_flow = od.demand
        for link in min_path.included_links:
            link.flow += od.demand
    net.update_all_link_cost()
    net.update_all_path_cost()
    # main loop
    iter_times, cur_gap = 0, inf
    while cur_gap >= lower_problem_gap:
        iter_times += 1
        # Step 2: Direction finding
        for od in net.OD:
            min_dist, min_path = od.find_shortest_path_in_PPS()
            if min_path not in od.working_path_set:
                od.working_path_set.append(min_path)
            # Step 3: Flow shift between paths
            for p in od.working_path_set:
                if p == min_path:
                    continue
                temp = sum([link.derivative() for link in list(set(p.included_links) ^ set(min_path.included_links))])
                shifted_flow = min((p.path_cost - min_dist) / temp, p.path_flow)
                # Everytime a flow shift happens, update cost
                p.path_flow -= shifted_flow
                for link in p.included_links:
                    link.flow -= shifted_flow
                    link.update_ue_cost()
                min_path.path_flow += shifted_flow
                for link in min_path.included_links:
                    link.flow += shifted_flow
                    link.update_ue_cost()
                net.update_all_path_cost()
                # Step 4: Update the working path set
            od.working_path_set = [path for path in od.working_path_set if path.path_flow != 0]
        # Step 5: Convergence test
        SPTT = 0
        for od in net.OD:
            min_dist, min_path = od.find_shortest_path_in_PPS()
            SPTT += min_dist * od.demand
        TSTT = sum([link.flow * link.cost for link in net.LINKS[1:]])
        cur_gap = (TSTT / SPTT) - 1
    return sum([link.flow * link.cost for link in net.LINKS[1:]])


def solutions_enumeration(net: Network, log_out=False):
    # Enumeration for the optimal path information provision strategy
    file_name = f'results\\Solution_Enumeration_{datetime.strftime(datetime.now(), "%y%m%d_%H%M%S")}.txt'
    if log_out:
        with open(file_name, 'a') as file:
            file.write(f"Enumeration Test\nNetwork={net.name}\n\n")
    min_tstt, optimal_solution = inf, None
    solutions = product(range(min_path_number, max_path_number+1), repeat=len(net.OD))
    for solution in solutions:
        tstt = lower_problem(net, ppn=solution)
        if tstt < min_tstt:
            min_tstt, optimal_solution = tstt, solution
            print(f"A new minimal TSTT has occurred:\npath={solution}\ntstt={tstt}")
            if log_out:
                with open(file_name, 'a') as file:
                    file.write(f"A new minimal TSTT has occurred:\npath={solution}\ntstt={tstt}\n")
    return optimal_solution, min_tstt


class SimulatedAnnealing:
    def __init__(self, net, initial_solution=None):
        self.net: Network = net
        self.initial_solution = initial_solution

    @staticmethod
    def single_point_disruption(ppn):
        new_ppn = ppn.copy()
        loc = np.random.randint(low=0, high=len(ppn))
        if new_ppn[loc] == max_path_number:
            new_ppn[loc] -= 1
        elif new_ppn[loc] == min_path_number:
            new_ppn[loc] += 1
        else:
            new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    @staticmethod
    def multi_point_disruption(ppn):
        new_ppn = ppn.copy()
        locs = np.random.randint(low=0, high=len(ppn), size=multi_num)
        for loc in locs:
            if new_ppn[loc] == max_path_number:
                new_ppn[loc] -= 1
            elif new_ppn[loc] == min_path_number:
                new_ppn[loc] += 1
            else:
                new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    def block_disruption(self, ppn):
        new_ppn = ppn.copy()
        left = np.random.randint(low=0, high=len(ppn))
        right = min(len(self.net.OD), left+block_num)
        for loc in range(left, right):
            if new_ppn[loc] == max_path_number:
                new_ppn[loc] -= 1
            elif new_ppn[loc] == min_path_number:
                new_ppn[loc] += 1
            else:
                new_ppn[loc] += random.choice([-1, 1])
        return new_ppn

    def run(self):
        # initialize the parameters
        if self.initial_solution:
            cur_solution = self.initial_solution
        else:
            cur_solution = [initial_num] * len(self.net.OD)
        cur_temperature = initial_temp
        cur_objective = lower_problem(self.net, cur_solution)
        best_solution = cur_solution.copy()
        best_objective = cur_objective
        obj = [best_objective]
        # log out the information
        cur_time = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
        directories = f"results\\SA{cur_time}\\solution_lists"
        os.makedirs(directories)
        with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
            file.write(f"Simulation Annealing Process\nInitial temperature={initial_temp}\n")
            file.write(f"Cooling rate={initial_cooling_rate, later_cooling_rate}"
                       f"\nMarkov chain length={markov_chain_length}\n")
            file.write(f"Total iteration times={num_iteration}\nNetwork {self.net.name}\n\n")
            file.write(f"Number of block disruption={block_num}\nmulti-point disruption={multi_num}\n\n")
        # main loop
        iter_times = 1
        non_improve_times = 0
        s = time.perf_counter()
        while True:
            if iter_times % markov_chain_length == 0:
                print(f'iteration {iter_times}, temperature={cur_temperature}, cur_obj={cur_objective}, best_obj={best_objective}')
            # find a neighborhood
            if cur_temperature > 0.5 * initial_temp:  # initial stage
                candidate = self.block_disruption(cur_solution)
            elif cur_temperature > 0.2 * initial_temp:  # middle stage
                candidate = self.multi_point_disruption(cur_solution)
            else:  # later stage
                candidate = self.single_point_disruption(cur_solution)
            candidate_objective = lower_problem(self.net, ppn=candidate)
            # search
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
                else:
                    non_improve_times += 1
            elif np.random.uniform(low=0, high=1) < np.exp((cur_objective - candidate_objective) / cur_temperature):
                cur_solution = candidate.copy()
                cur_objective = candidate_objective
                non_improve_times += 1
            else:
                non_improve_times += 1
            # cooling the temperature
            if iter_times % markov_chain_length == 0:
                if iter_times <= num_iteration * 0.5:
                    cur_temperature *= initial_cooling_rate
                else:
                    cur_temperature *= later_cooling_rate
            if non_improve_times == max_non_improved_times:
                print(f"Reach the maximum non-improve times")
                e = time.perf_counter()
                with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
                    file.write(f"Algorithm terminates, time usage = {e - s:.2f}s.")
                return best_solution, best_objective, obj
            if cur_temperature <= min_temperature:
                print(f"The temperature has dropped to {min_temperature}")
                e = time.perf_counter()
                with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
                    file.write(f"Algorithm terminates, time usage = {e - s:.2f}s.")
                return best_solution, best_objective, obj
            if iter_times == num_iteration:
                e = time.perf_counter()
                with open(f"results\\SA{cur_time}\\iteration.txt", "a") as file:
                    file.write(f"Algorithm terminates, time usage = {e - s:.2f}s.")
                return best_solution, best_objective, obj
            iter_times += 1


def main():
    sf = Network(name="SiouxFalls", sst=1, model="UE", from_file=True)
    sf.generate_total_path_set()
    with open(r'results\SA250331_124031\solution_lists\9282iter.pkl', 'rb') as f:
        ppn = pickle.load(f)
    # print(lower_problem(sf, ppn))
    # new_ppn = [[0, 0, 0, 0, 0] for _ in range(len(sf.OD))]
    # for i, od in enumerate(sf.OD):
    #     for j, path in enumerate(od.potential_path_set):
    #         if path.path_flow != 0:
    #             new_ppn[i][j] = 1
    # print(new_ppn)
    # print(lower_problem(sf, new_ppn))
    sa = SimulatedAnnealing(sf, ppn)
    sa.run()


if __name__ == '__main__':
    main()
