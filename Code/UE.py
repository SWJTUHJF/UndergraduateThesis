"""
User equilibrium solved by Frank-Wolfe algorithm.
Network: ['Anaheim', 'Austin', 'Barcelona', 'Berlin-Center', 'Berlin-Friedrichshain', 'Berlin-Mitte-Center',
          'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center', 'Berlin-Prenzlauerberg-Center', 'Berlin-Tiergarten',
          'Birmingham-England', 'br12', 'Braess-Example', 'chicago-regional', 'Chicago-Sketch',
          'Eastern-Massachusetts', 'GoldCoast', 'Hessen-Asymmetric', 'Munich', 'Philadelphia', 'SiouxFalls',
          'Sydney', 'SymmetricaTestCase', 'Terrassa-Asymmetric', 'Winnipeg', 'Winnipeg-Asymmetric']
Algorithm name: GLC(Bellman-Ford), LC(SPFA), LS(Dijkstra), LSF(Dijkstra with priority queue).
Algorithm efficiency: LSF >= LS > LC > GLC
"""


import heapq
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from datetime import datetime
from math import inf


class NODE:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.upstream_link: list[LINK] = list()
        self.downstream_link: list[LINK] = list()
        self.p: NODE = self
        self.u: float = inf
        self.visited: bool = False  # For SPFA algorithm

    def __repr__(self):
        return f'NODE {self.node_id}'


class LINK:
    def __init__(self, link_id, tail=None, head=None, capacity=None,
                 length=None, free_flow_time=None, alpha=None, beta=None):
        self.link_id: int = link_id
        self.tail: NODE = tail
        self.head: NODE = head
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = free_flow_time
        self.alpha: float = alpha
        self.beta: float = beta
        self.flow: float = 0
        self.auxiliary_flow: float = 0
        self.cost: float = 0
        if link_id != 0:
            self.update_cost()

    def __repr__(self):
        return f'LINK {self.link_id} cost = {self.cost}, flow = {self.flow}'

    def update_cost(self, model="UE"):
        if model == "UE":
            self.cost = self.fft * (1 + self.alpha * (self.flow / self.capacity) ** self.beta)
        elif model == "SO":
            self.cost = (self.fft * (1 + self.alpha * (self.flow / self.capacity) ** self.beta)
                         + self.fft * self.alpha * self.beta * (self.flow / self.capacity) ** self.beta)
        else:
            raise ValueError()

    def get_specific_cost(self, temp, model='UE'):
        if model == "UE":
            return self.fft * (1 + self.alpha * (temp / self.capacity) ** self.beta)
        elif model == "SO":
            return (self.fft * (1 + self.alpha * (temp / self.capacity) ** self.beta)
                    + self.fft * self.alpha * self.beta * (temp / self.capacity) ** self.beta)
        else:
            raise ValueError()


class ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: NODE = origin
        self.destination: NODE = destination
        self.demand: float = demand

    def __repr__(self):
        return f'ODPair {self.origin.node_id}->{self.destination.node_id}={self.demand}'


class FW:
    def __init__(self, name, algorithm, BC, FWC, model="UE", sst=1):
        self.name: str = name
        self.alg_name = algorithm
        self.alg = self.get_algorithm(algorithm)
        self.BC: float = BC  # bisection method convergence
        self.FWC: float = FWC  # Frank-Wolfe algorithm convergence
        self.model: str = model
        self.NODES: list[NODE] = list()
        self.LINKS: list[LINK] = list()
        self.ODPAIRS: list[ODPair] = list()
        self.non: int = 0
        self.gap_list: list = list()
        self.run_time: float = 0
        self.sst = sst
        self.main()

    """
    Part0: Miscellaneous
    """
    def main(self):
        dir1 = "TransportationNetworks"
        dir2, dir3_net, dir3_od = self.name, None, None
        files = [name for name in os.listdir(f"{dir1}\\{dir2}")]
        for f in files:
            if "net" in f:
                dir3_net = f
            if "trips" in f:
                dir3_od = f
        self.read_network(f'{dir1}\\{dir2}\\{dir3_net}')
        self.read_OD(f'{dir1}\\{dir2}\\{dir3_od}')

    def get_algorithm(self, alg_name):
        collection = {"GLC": self.GLC,
                      "LC": self.LC,
                      "LS": self.LS,
                      "LSF": self.LSF}
        return collection[alg_name]

    """
    Part1: read the network, create instances.
    """
    def read_network(self, path):
        with open(path, 'r', encoding='UTF-8') as f1:
            # Process the text file
            lines = f1.readlines()
            pattern = re.compile(r'[\w.~]+')
            data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
            self.non = int(data[1][-1])
            for i in range(len(data)):
                if '~' in data[i] and "ORIGINAL" not in data[i]:
                    data = data[i + 1:]
                    break
            # Create NODE and LINK object
            self.NODES = [NODE(i) for i in range(self.non + 1)]  # Be CAREFUL that position 0 represents nothing
            self.LINKS = [LINK(0)]
            for index, line in enumerate(data):
                temp = LINK(index + 1, self.NODES[int(line[0])], self.NODES[int(line[1])], float(line[2]),
                            float(line[3]), float(line[4]), float(line[5]), float(line[6]))
                self.LINKS.append(temp)
                self.NODES[int(line[0])].downstream_link.append(temp)
                self.NODES[int(line[1])].upstream_link.append(temp)

    def read_OD(self, path):
        with open(path, 'r', encoding='UTF-8') as f1:
            # Process the text file
            lines = f1.readlines()
            pattern = re.compile(r'[0-9.]+|Origin')
            data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
            total_flow = float(data[1][0])
            for i in range(len(data)):
                if 'Origin' in data[i]:
                    data = data[i:]
                    break
            # Create NODE and LINK object
            for line in data:
                if "Origin" in line:
                    origin = self.NODES[int(line[-1])]
                else:
                    for i in range(len(line) // 2):
                        destination = self.NODES[int(line[2 * i])]
                        demand = float(line[2 * i + 1]) * self.sst
                        if demand != 0:
                            self.ODPAIRS.append(ODPair(origin, destination, demand))
            # Check the correctness of OD flows
            # if abs(total_flow - sum([od.demand for od in self.ODPAIRS])) > 1:
            #     raise ValueError("Data in the file does not match with the total OD flow.")

    """
    Part2: define various shortest path algorithm
    """
    def initialize(self, o_id):
        # Initialize parameters
        for node in self.NODES:
            node.p = node
            node.u = inf
            node.visited = False
        self.NODES[o_id].u = 0
        self.NODES[o_id].p = -1
        self.NODES[o_id].visited = True

    # Based on the current parent label, obtain the shortest path.
    def obtain_shortest_path(self, d_id: int):
        shortest_path_links = list()
        current_node = self.NODES[d_id]
        while current_node.p != -1:
            for link in current_node.upstream_link:
                if link.tail == current_node.p:
                    shortest_path_links.append(link)
                    break
            else:
                raise ValueError('Something is wrong with shortest path.')
            current_node = current_node.p
        return shortest_path_links[::-1]

    # GLC or Bellman-Ford algorithm
    def GLC(self, o_id: int, d_id: int):
        self.initialize(o_id)
        for _ in self.NODES[2:]:
            updated = False
            for link in self.LINKS[1:]:
                tail_node, head_node = link.tail, link.head
                if tail_node.u != inf and head_node.u > tail_node.u + link.cost:
                    head_node.u = tail_node.u + link.cost
                    head_node.p = tail_node
                    updated = True
            if not updated:
                break

    # LC or SPFA algorithm
    def LC(self, o_id: int, d_id: int):
        self.initialize(o_id)
        SEL = deque([self.NODES[o_id]])
        while len(SEL):
            cur_node = SEL.popleft()
            cur_node.visited = False
            for link in cur_node.downstream_link:
                next_node = link.head
                if cur_node.u != inf and next_node.u > cur_node.u + link.cost:
                    next_node.u = cur_node.u + link.cost
                    next_node.p = cur_node
                    if not next_node.visited:
                        SEL.append(next_node)

    # LS or Dijkstra algorithm
    def LS(self, o_id: int, d_id: int):
        self.initialize(o_id)
        SEL = [self.NODES[o_id]]
        while SEL:
            SEL.sort(key=lambda node: node.u, reverse=True)
            cur_node = SEL.pop()
            if cur_node == self.NODES[d_id]:
                break
            for link in cur_node.downstream_link:
                next_node, dist = link.head, link.cost
                if next_node.u > cur_node.u + dist:
                    next_node.u = cur_node.u + dist
                    next_node.p = cur_node
                    if next_node not in SEL:
                        SEL.append(next_node)

    # Dijkstra priority queue algorithm
    def LSF(self, o_id: int, d_id: int):
        self.initialize(o_id)
        SEL = list()
        heapq.heappush(SEL, (self.NODES[o_id].u, o_id))
        while SEL:
            cur_dist, cur_node = heapq.heappop(SEL)
            cur_node = self.NODES[cur_node]
            if cur_node == self.NODES[d_id]:
                break
            for link in cur_node.downstream_link:
                next_node, dist = link.head, link.cost
                if next_node.u > cur_node.u + dist:
                    next_node.u = cur_node.u + dist
                    next_node.p = cur_node
                    if (next_node.u, next_node.node_id) not in SEL:
                        heapq.heappush(SEL, (next_node.u, next_node.node_id))

    """
    Part3: Frank-Wolfe algorithm
    """
    def update_costs(self):
        for link in self.LINKS[1:]:
            link.update_cost(self.model)

    def update_flow(self, step):
        for link in self.LINKS[1:]:
            link.flow = link.flow + step * (link.auxiliary_flow - link.flow)

    def all_or_nothing(self):
        for link in self.LINKS:
            link.auxiliary_flow = 0
        for od in self.ODPAIRS:
            ori = od.origin
            dest = od.destination
            demand = od.demand
            self.alg(ori.node_id, dest.node_id)
            shortest_links = self.obtain_shortest_path(dest.node_id)
            for link in shortest_links:
                link.auxiliary_flow += demand

    def derivative_f(self, alpha):
        return sum([link.get_specific_cost(link.flow + alpha * (link.auxiliary_flow - link.flow), model=self.model)
                    * (link.auxiliary_flow - link.flow) for link in self.LINKS[1:]])

    def bisection(self):
        left, right, mid = 0, 1, 0.5
        max_iter_times = 500
        iter_times = 1
        while abs(self.derivative_f(mid)) > self.BC:
            iter_times += 1
            if iter_times == max_iter_times:
                raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
            elif self.derivative_f(mid) * self.derivative_f(right) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
        return mid

    def convergence(self):
        SPTT = 0
        for od in self.ODPAIRS:
            self.LS(od.origin.node_id, od.destination.node_id)
            min_path = self.obtain_shortest_path(od.destination.node_id)
            min_dist = sum([link.cost for link in min_path])
            SPTT += min_dist * od.demand
        TSTT = sum([link.flow * link.cost for link in self.LINKS[1:]])
        cur_gap = (TSTT / SPTT) - 1
        return cur_gap

    def conduct_FW(self):
        start = time.perf_counter()
        iter_times = 0
        cur_gap = inf
        self.all_or_nothing()
        for link in self.LINKS:
            link.flow = link.auxiliary_flow
        while cur_gap > self.FWC:
            self.update_costs()
            self.all_or_nothing()
            step = self.bisection()
            self.update_flow(step)
            cur_gap = self.convergence()
            iter_times += 1
            self.gap_list.append(cur_gap)
        end = time.perf_counter()
        self.run_time = end - start

    """
    Part4: log out information
    """
    def total_system_travel_time(self):
        total = sum([link.get_specific_cost(link.flow, model="UE") * link.flow for link in self.LINKS[1:]])
        print(f'Total system travel time is {total:.2f}')
        return total

    def print_gap(self):
        for i, gap in enumerate(self.gap_list):
            if i % 5 == 0:
                print(f'iteration {i}: gap = {gap:.5f}')

    def plot_gap(self, save=False, path=None):
        x = list(range(len(self.gap_list)))
        y = np.array(self.gap_list)
        fig, ax = plt.subplots()
        ax.plot(x, y, linestyle='-', color='g', label=f'UE-{self.name}-{self.alg_name}')
        ax.set_yticks([10**i for i in range(-4, 1)])
        ax.set_yscale('log')
        ax.set_xlabel('Iteration Time')
        ax.set_ylabel('Relative Gap')
        ax.legend()
        ax.grid(True)
        if save:
            plt.savefig(f'{path}/relative_gap.png', dpi=300, bbox_inches='tight')

    def log_out(self):
        dir_name = f"{self.name}_{self.alg_name}_{datetime.now().strftime('%y%m%d%H%M%S')}"
        os.mkdir(f'Results/{dir_name}')
        file_path = f'Results/{dir_name}/Data.txt'
        with open(file_path, 'w') as f:
            f.write(f"***Data for User Equilibrium***\n"
                    f"network: {self.name}\n"
                    f"SSP algorithm: {self.alg_name}\n"
                    f"iteration gap: {self.FWC}\n"
                    f"Total expected travel time: {self.total_system_travel_time():.2f}\n"
                    f"Total running time: {self.run_time:.2f}s\n"
                    f"Flow and cost of each link:\n")
            for link in self.LINKS[1:]:
                f.write(f"link {link.link_id}: flow = {link.flow:.2f}, cost = {link.cost:.2f}\n")
        self.plot_gap(save=True, path=f'Results/{dir_name}')
        print(f"Total running time {self.run_time:.2f}s")
        print(f"Results are stored at Results/{dir_name}")


def list_network_names():
    net_names = [name for name in os.listdir("TransportationNetworks")
                 if 97 <= ord(name[0]) <= 122 or 65 <= ord(name[0]) <= 90 and '.md' not in name]
    return net_names


if __name__ == "__main__":
    net = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="UE", sst=1)
    net.conduct_FW()
    net.total_system_travel_time()
