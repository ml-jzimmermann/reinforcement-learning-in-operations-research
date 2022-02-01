from warehouse_v7 import WarehouseV7
import time
from tqdm import trange
import os
import math


def clear():
    os.system('clear')


# define env sizes
small = {
    'num_aisles': 1,
    'rack_height': 3,
    'min_packets': 4,
    'max_packets': 4,
    'seed': 3
}

medium = {
    'num_aisles': 2,
    'rack_height': 5,
    'min_packets': 4,
    'max_packets': 4,
    'seed': 3
}

big = {
    'num_aisles': 4,
    'rack_height': 8,
    'min_packets': 4,
    'max_packets': 4,
    'seed': 3
}

config = big

# setup matching warehouse env
warehouse = WarehouseV7(num_aisles=config['num_aisles'], rack_height=config['rack_height'],
                        min_packets=config['min_packets'], max_packets=config['max_packets'],
                        seed=config['seed'])


# return policy
def return_policy():
    steps = 0

    width = warehouse.width
    height = warehouse.height
    num_aisles = warehouse.num_aisles + 2
    packets = warehouse.get_packets()

    # walk the cross aisle twice
    steps += width * 2

    # iterate over aisles to collect packets
    for aisle in range(num_aisles):
        aisle_index = aisle * 3

        # if packets in aisle, go get them
        max_packet_index = 0
        for p in packets:
            if p[0] in [aisle_index - 1, aisle_index + 1]:
                packet_index = height - p[1]
                if packet_index > max_packet_index:
                    max_packet_index = packet_index

        # walk up and down to aisle
        steps += max_packet_index * 2

    return steps


# transversal policy
def transversal_policy():
    steps = 0

    width = warehouse.width
    height = warehouse.height
    num_aisles = warehouse.num_aisles + 2

    # walk up and down
    steps += height * num_aisles
    steps += 3 * (num_aisles - 1)

    # get back home
    if num_aisles % 2 == 0:
        steps += width
    else:
        steps += width + height
    return steps


# midpoint policy
def midpoint_policy():
    steps = 0

    width = warehouse.width
    height = warehouse.height
    num_aisles = warehouse.num_aisles + 2
    rack_height = warehouse.rack_height
    packets = warehouse.get_packets()

    # define midpoint
    midpoint = math.floor(rack_height / 2)

    # walk under midpoint
    steps += width

    # iterate over aisles to collect packets
    for aisle in range(num_aisles - 1):
        aisle_index = aisle * 3

        # if packets in aisle, go get them
        min_packet_index = height
        for p in packets:
            if p[0] in [aisle_index - 1, aisle_index + 1] and midpoint < p[1] < min_packet_index:
                min_packet_index = p[1]

        # walk up and down to aisle
        steps += (height - min_packet_index) * 2

    # walk over midpoint
    steps += width + height

    # iterate over aisles to collect packets
    for aisle in reversed(range(1, num_aisles)):
        aisle_index = aisle * 3

        # if packets in aisle, go get them
        max_packet_index = 0
        for p in packets:
            if p[0] in [aisle_index - 1, aisle_index + 1] and midpoint >= p[1] > max_packet_index:
                max_packet_index = p[1]

        # walk up and down to aisle
        steps += max_packet_index * 2

    # walk home
    steps += height

    return steps


total_episodes = 100000
return_steps = 0
transversal_steps = 0
midpoint_steps = 0

for episode in trange(total_episodes):
    warehouse.reset()
    return_steps += return_policy()
    transversal_steps += transversal_policy()
    midpoint_steps += midpoint_policy()

print(f'return policy steps average: {return_steps / total_episodes:.2f}')
print(f'transversal policy steps average: {transversal_steps / total_episodes:.2f}')
print(f'midpoint policy steps average: {midpoint_steps / total_episodes:.2f}')
