from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch as T
import time
import os

from warehouse_v7 import WarehouseV7


def clear():
    os.system('clear')


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_features=input_size, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        f1 = F.relu(self.l1(x))
        f2 = F.relu(self.l2(f1))
        out = self.out(f2)
        return out, f1, f2


big = T.load('../../../models/final/pytorch_warehouse_v7_thesis_big_less_dqn_7_20.12_14.35_50000.pt',
             map_location=T.device('cpu'))
small = T.load('../../../models/final/pytorch_warehouse_v7_thesis_small_dqn_11_18.12_15.07_50000.pt',
               map_location=T.device('cpu'))
medium = T.load('../../../models/final/pytorch_warehouse_v7_thesis_medium_dqn_7_18.12_15.21_50000.pt',
                map_location=T.device('cpu'))
# checkpoints = {'klein': small, 'mittel': medium, 'groß': big}
checkpoints = {'groß': big}
colors = {'klein': 'red', 'mittel': 'blue', 'groß': 'purple'}
fig = plt.figure(1, figsize=(8, 4))

for label, checkpoint in checkpoints.items():
    warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'],
                            rack_height=checkpoint['config']['rack_height'],
                            min_packets=checkpoint['config']['min_packets'],
                            max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'],
                            center_target=False)

    action_labels = warehouse.get_action_labels()

    model = Model(input_size=warehouse.observation_space, output_size=warehouse.action_space)
    model.load_state_dict(checkpoint['policy_model_state_dict'])

    max_q_values = []

    episodes = 1
    episode = 0
    max_steps = 100

    print('generating observations...')
    for i in tqdm(range(1000)):
        episode_q_values = []

        state, info = warehouse.reset(include_info=True)

        done = False
        steps = 0

        while not done:
            action, f1, f2 = model(T.tensor(state).float().unsqueeze(0))
            action, f1, f2 = action.detach().squeeze().numpy(), f1.detach().squeeze().numpy(), f2.detach().squeeze().numpy()
            episode_q_values.append(max(action))

            steps += 1
            state, reward, done, info = warehouse.step(np.argmax(action))
            clear()
            warehouse.render()
            time.sleep(0.1)

            if steps >= max_steps:
                episode += 1
                max_q_values.append(episode_q_values)
                break

            if done:
                ...

        if episode >= episodes:
            break

    for q_values in max_q_values:
        plt.plot(range(0, len(q_values)), q_values, c=colors[label], linewidth=1, label=label)

plt.ylabel('Q-Wert')
plt.xlabel('Schritt')
plt.tight_layout()
# plt.legend()
plt.show()
