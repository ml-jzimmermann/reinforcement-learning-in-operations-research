from src.model import DQN, LstmDQN
from src.agent import QAgent, RecurrentQAgent
from src.warehouse.environment.warehouse_v7 import WarehouseV7

import torch

checkpoint = torch.load('../../models/pytorch_warehouse_v7_bigger_lstm_03.02_16.44_100000.pt',
                        map_location=torch.device('cpu'))
warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'])

checkpoint['config']['device'] = 'cpu'

agent = RecurrentQAgent(env=warehouse, model=LstmDQN, config=checkpoint['config'])
agent.compile()

agent.load_checkpoint(checkpoint)
# agent.plot_durations()

# print(checkpoint['config'])

for _ in range(10):
    agent.play(verbose=True, sleep=0.1, max_steps=150, include_q_values=True)
