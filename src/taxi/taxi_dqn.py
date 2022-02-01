import torch

from policy import ExponentialEpsilonGreedyPolicy
from src.model import EmbeddingDQN
from src.agent import QAgent
import gym
import torch.nn.functional as F
import torch.optim as optimizer

# definition of hyperparameters
hyperparameters = {
    # training
    'batch_size': 256,
    'learning_rate': 0.001,
    'scheduler_milestones': [5000],
    'scheduler_decay': 0.1,
    'optimizer': optimizer.Adam,
    'loss': F.smooth_l1_loss,
    'running_k': 4,
    'bidirectional': False,
    'combined_memory': False,
    # reinforcement & environment
    'eps_policy': ExponentialEpsilonGreedyPolicy(eps_max=1.0, eps_min=0.02, decay=1000),
    'gamma': 0.99,
    'target_update': 20,
    'num_episodes': 10001,
    'memory_capacity': 50000,
    'warmup_episodes': 10,
    'save_freq': 2000,
    'max_steps_per_episode': 100,
    # pytorch
    'np_seed': 42,
    'device': 'cpu',
    'save_model': True,
    'dtype': torch.long,
    'plot_progress': True,
    'ylim': (-200, 200),
    'tag': f'taxi_v3'
}

# setup environment
taxi = gym.make('Taxi-v3').env
taxi.seed(42)

# setup and train the agent
agent = QAgent(env=taxi, model=EmbeddingDQN, config=hyperparameters)
agent.compile()
agent.fit()
