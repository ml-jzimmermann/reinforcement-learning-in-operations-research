import torch as T
import os
import json
from src.model import DQN
from src.agent import QAgent
from src.warehouse.environment.warehouse_v7 import WarehouseV7

filepath = '../../../models/final'
results = {}
for file in os.listdir(filepath):
    if 'big_less_centered' in file:
        checkpoint = T.load(f'{filepath}/{file}', map_location=T.device('cpu'))
        warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'],
                                rack_height=checkpoint['config']['rack_height'],
                                min_packets=checkpoint['config']['min_packets'],
                                max_packets=checkpoint['config']['max_packets'],
                                seed=checkpoint['config']['np_seed'],
                                center_target=True)

        agent = QAgent(env=warehouse, model=DQN, config=checkpoint['config'])
        agent.compile()

        agent.load_checkpoint(checkpoint)
        solved, reward, steps = agent.evaluate(num_episodes=100000)
        results['_'.join(file.split('_')[1:9])] = {'solved': solved, 'reward': reward, 'steps': steps}

with open('results_target_centered.txt', 'a') as output:
    output.write(json.dumps(results))
    output.write('\n')
    output.flush()
