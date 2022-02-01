from src.model import DQN, LstmDQN
from src.agent import QAgent, RecurrentQAgent
from src.warehouse.environment.warehouse_v7 import WarehouseV7

import torch

checkpoint = torch.load('../../models/v7/pytorch_warehouse_v7_big_lstmdqn_lr2_28.09_06.50_140000.pt',
                        map_location=torch.device('cpu'))
warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'])

checkpoint['config']['init_weights'] = False
checkpoint['config']['device'] = 'cpu'
checkpoint['config']['combined_memory'] = False
checkpoint['config']['bidirectional'] = False
checkpoint['config']['ylim'] = (-550, 250)

agent = RecurrentQAgent(env=warehouse, model=LstmDQN, config=checkpoint['config'])
agent.compile()

agent.load_checkpoint(checkpoint)
agent.plot_durations()

print(checkpoint['config'])

# for _ in range(10):
#     agent.play(verbose=True, sleep=3, max_steps=75, include_q_values=True)
