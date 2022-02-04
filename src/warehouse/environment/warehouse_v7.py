import random
import numpy as np
import math
from gym.spaces import Discrete


# needed to fix indices in rendering because the rendering appears wider as the actual warehouse
# the walls are printed as extra characters -> the agent has to skip them during horizontal travel
def _get_render_index(width):
    indices = {
        0: 2,
        1: 5,
        2: 6,
    }
    return (width // 3) * 7 + indices[width % 3]


class WarehouseV7:
    def __init__(self, rack_height=5, num_aisles=2, min_packets=3, max_packets=8, seed=None, center_target=False):
        super(WarehouseV7, self).__init__()
        # setup parameters
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.rack_height = rack_height
        self.num_aisles = num_aisles
        self.min_packets = min_packets
        self.max_packets = max_packets

        # env definition
        self.rack_width = 2
        self.aisle_width = 1
        self.height = rack_height + 2
        # calculate warehouse width based on number of aisles and rack width
        self.width = (num_aisles * (self.rack_width + self.aisle_width)) + 2 + self.rack_width
        self.num_racks = num_aisles + 1
        self.packets = []
        self.num_packets = 0
        self.steps_taken = 0
        self.packets_collected = 0
        self.packet_height_range = range(1, self.rack_height + 1)
        self.action_labels = {0: 'hoch', 1: 'rechts', 2: 'runter', 3: 'links'}
        width_range = []
        for rack in range(self.num_racks):
            width_range.append(rack * (self.rack_width + self.aisle_width) + 1)
            width_range.append(rack * (self.rack_width + self.aisle_width) + 2)
        self.packet_width_range = width_range

        # calculate target index from target position
        if center_target:
            self.target_index = math.floor(self.width / 2)
        else:
            self.target_index = 0

        # set agent position in front of target
        self.agent_position = (self.target_index, self.height - 1)

        # define dimensions
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.width * self.height)
        self.info = {'collected': 0, 'steps': 0, 'agent_position': self.agent_position}

    # check if certain position is visitable to give negative reward if the agent tries to run into walls or racks
    def _is_visitable(self, position):
        w = position[0]
        h = position[1]
        w_range = []
        for rack in range(self.num_racks):
            w_range.append(rack * (self.rack_width + self.aisle_width) + 0)
            w_range.append(rack * (self.rack_width + self.aisle_width) + 3)

        if h == 0 or h == (self.height - 1):
            return w in range(self.width)
        if h in range(1, self.height - 1) and w in w_range:
            return True
        return False

    # check if packages can be collected from current position and if so do it
    def _collect_packet_if_possible(self, position):
        w = position[0]
        h = position[1]
        num_collected = 0
        collected = []
        if h in range(1, self.height - 1):
            for index, packet in enumerate(self.packets):
                if (packet[0] == w + 1 or packet[0] == w - 1) and packet[1] == h:
                    collected.append(packet)
                    self.packets_collected += 1
                    num_collected += 1
        for c in collected:
            self.packets.remove(c)
        return num_collected

    # agent is in front of deposit
    def _has_deposit_in_reach(self, position):
        return position[0] == self.target_index and position[1] == self.height - 1

    def get_action_label(self, action):
        return self.action_labels[action]

    def get_action_labels(self):
        return self.action_labels

    # perform the action if allowed and return rewards, info, terminal signal and new observations
    def step(self, action):
        done = False

        if action < 0 or action > 3:
            raise ValueError("Invalid action")

        if action == 0:
            new_position = (self.agent_position[0], self.agent_position[1] - 1)

        if action == 1:
            new_position = (self.agent_position[0] + 1, self.agent_position[1])

        if action == 2:
            new_position = (self.agent_position[0], self.agent_position[1] + 1)

        if action == 3:
            new_position = (self.agent_position[0] - 1, self.agent_position[1])

        if self._is_visitable(new_position):
            reward = -1
            self.agent_position = new_position
            num_collected = self._collect_packet_if_possible(new_position)
            if self.packets_collected < self.num_packets and num_collected > 0:
                reward = 10 * num_collected

            if self.packets_collected == self.num_packets and self._has_deposit_in_reach(new_position):
                done = True
                reward = 100
        else:
            reward = -5

        next_state = self._generate_observations()
        self.steps_taken += 1
        self.info['agent_position'] = self.agent_position
        return next_state, reward, done, self.info

    def get_packets(self):
        return self.packets

    # generate a new iteration: randomly distribute packages and reset all values
    def reset(self, include_info=False):
        self.agent_position = (self.target_index, self.height - 1)
        self.packets_collected = 0
        self.steps_taken = 0
        self.packets.clear()
        packets = set()
        self.num_packets = np.random.randint(low=self.min_packets, high=self.max_packets + 1)
        self.info = {'collected': 0, 'steps': 0, 'packages': self.num_packets, 'agent_position': self.agent_position}
        while len(packets) < self.num_packets:
            packets.add((random.choice(self.packet_width_range), random.choice(self.packet_height_range)))
        for p in packets:
            self.packets.append(p)

        if include_info:
            return self._generate_observations(), self.info
        else:
            return self._generate_observations()

    # generating observation matrix
    def _generate_observations(self):
        observations = np.zeros((self.height, self.width))
        for p in self.packets:
            observations[p[1], p[0]] = 1
        observations[self.agent_position[1], self.agent_position[0]] = 1
        if len(self.packets) == 0:
            observations[self.height - 1, self.target_index] = 1
        return observations.flatten()

    # render the observation for human review
    # q-values can be included
    def render(self, current_q_values=None):
        render_string = ''
        render_row = '_' * ((self.num_aisles + 2) * 5 + (self.num_racks * 2)) + '\n'
        render_string += render_row
        agent_index = _get_render_index(self.agent_position[0])
        render_row = '|' + '    __ ' * self.num_racks + '   |' + '\n'
        if self.agent_position[1] == 0:
            render_row = render_row[:agent_index - 1] + ' A ' + render_row[agent_index + 2:]
        render_string += render_row

        for row in range(1, self.rack_height):
            render_row = '|' + '   |  |' * self.num_racks + '   |' + '\n'
            if self.agent_position[1] == row:
                render_row = render_row[:agent_index] + 'A' + render_row[agent_index + 1:]
            for packet in self.packets:
                if packet[1] == row:
                    packet_index = _get_render_index(packet[0])
                    render_row = render_row[:packet_index] + 'P' + render_row[packet_index + 1:]

            render_string += render_row

        render_row = '|' + '   |__|' * self.num_racks + '   |' + '\n'
        if self.agent_position[1] == self.rack_height:
            render_row = render_row[:agent_index] + 'A' + render_row[agent_index + 1:]
        for packet in self.packets:
            if packet[1] == self.rack_height:
                packet_index = _get_render_index(packet[0])
                render_row = render_row[:packet_index] + 'P' + render_row[packet_index + 1:]
        render_string += render_row

        render_row = '|' + '_' * ((self.num_aisles + 2) * 5 + (self.num_racks * 2) - 2) + '|' + '\n'
        if self.agent_position[1] == self.rack_height + 1:
            render_row = render_row[:agent_index - 1] + ' A ' + render_row[agent_index + 2:]
        render_string += render_row
        render_string += ' ' * (_get_render_index(self.target_index) - 1) + '|D|'
        render_string += '\n'

        if current_q_values is not None:
            q_string = 'Q-Values: |'
            for i, q_value in enumerate(current_q_values):
                q_string += f' {self.get_action_label(i).upper()}: {q_value:6.2f} |'
            q_string += '\n'
            render_string += q_string

        # better agent including wider ailes
        render_string = render_string.replace(' A ', ' 🤖')
        render_string = render_string.replace('| |', '|   |')

        # better packages
        render_string = render_string.replace('P|', '📦')
        render_string = render_string.replace('|P', '📦')

        # better deposit
        render_string = render_string.replace('|D|', ' 🛒')

        print(render_string)
