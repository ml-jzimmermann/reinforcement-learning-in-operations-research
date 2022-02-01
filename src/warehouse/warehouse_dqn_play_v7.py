from src.model import DQN
from src.agent import QAgent
from src.warehouse.environment.warehouse_v7 import WarehouseV7

import torch

checkpoint = torch.load('../../models/final/pytorch_warehouse_v7_thesis_medium_dqn_11_18.12_16.16_50000.pt',
                        map_location=torch.device('cpu'))
warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'],
                        center_target=False)

agent = QAgent(env=warehouse, model=DQN, config=checkpoint['config'])
agent.compile()

agent.load_checkpoint(checkpoint)
agent.plot_durations()

for _ in range(10):
    agent.play(verbose=True, sleep=10, max_steps=75, include_q_values=True)

# agent.evaluate(num_episodes=10000)
