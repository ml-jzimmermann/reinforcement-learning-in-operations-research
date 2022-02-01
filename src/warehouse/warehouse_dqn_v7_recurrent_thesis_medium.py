from src.warehouse.environment.warehouse_v7 import WarehouseV7
from src.model import LstmDQN
from src.policy import ExponentialEpsilonGreedyPolicy
from src.agent import RecurrentQAgent

import torch
import torch.nn.functional as F
import torch.optim as optimizer

seeds = [3, 5, 7, 11, 13]

for seed in seeds:
    hyperparameters = {
        # training
        'batch_size': 32,
        'learning_rate': 0.001,
        'scheduler_milestones': [30000, 60000],
        'scheduler_decay': 0.1,
        'optimizer': optimizer.Adam,
        'loss': F.smooth_l1_loss,
        'running_k': 4,
        'bidirectional': False,
        'combined_memory': False,
        # reinforcement & environment
        'eps_policy': ExponentialEpsilonGreedyPolicy(eps_max=1.0, eps_min=0.02, decay=2000),
        'gamma': 0.9,
        'target_update': 10,
        'num_episodes': 50001,
        'memory_capacity': 50000,
        'warmup_episodes': 100,
        'save_freq': 25000,
        'max_steps_per_episode': 75,
        'num_aisles': 2,
        'rack_height': 5,
        'min_packets': 4,
        'max_packets': 4,
        # pytorch
        'np_seed': seed,
        'device': 'cpu',
        'save_model': True,
        'dtype': torch.float32,
        'plot_progress': False,
        'ylim': (-200, 200),
        'tag': f'warehouse_v7_thesis_medium_{seed}'
    }

    warehouse = WarehouseV7(num_aisles=hyperparameters['num_aisles'], rack_height=hyperparameters['rack_height'],
                            min_packets=hyperparameters['min_packets'], max_packets=hyperparameters['max_packets'],
                            seed=hyperparameters['np_seed'])

    agent = RecurrentQAgent(env=warehouse, model=LstmDQN, config=hyperparameters)
    agent.compile()
    agent.fit()
