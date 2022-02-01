from tqdm import tqdm
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch as T

from warehouse_v7 import WarehouseV7


# code to extract outputs of different layers of the network
# to later visualize using t-SNE / PCA


# modified version of the same network used in training that takes the same weights but outputs tensors for each layer
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # same structure as in models.py to match weight matrices
        self.l1 = nn.Linear(in_features=input_size, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        f1 = F.relu(self.l1(x))
        f2 = F.relu(self.l2(f1))
        out = self.out(f2)
        return out, f1, f2 # added intermediate layer outputs


# checkpoint to create matching warehouse env and load model weights
checkpoint = T.load('../models/final/pytorch_warehouse_v7_thesis_medium_dqn_5_18.12_14.33_50000.pt',
                    map_location=T.device('cpu'))

warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'],
                        center_target=False)

action_labels = warehouse.get_action_labels()

model = Model(input_size=warehouse.observation_space, output_size=warehouse.action_space)
model.load_state_dict(checkpoint['policy_model_state_dict'])

action_list = []
f1_list = []
f2_list = []
state_list = []
agent_pos_list = []

episodes = 75
total_points = 250
max_steps = 70

print('generating observations...')
for episode in tqdm(range(episodes)):
    episode_actions = []
    episode_f1 = []
    episode_f2 = []
    episode_agent_pos = []
    episode_states = []

    state, info = warehouse.reset(include_info=True)

    done = False
    steps = 0

    while not done:
        # perform action and extract the features
        action, f1, f2 = model(T.tensor(state).float().unsqueeze(0))
        action, f1, f2 = action.detach().squeeze().numpy(), f1.detach().squeeze().numpy(), f2.detach().squeeze().numpy()
        episode_actions.append(action)
        episode_f1.append(f1)
        episode_f2.append(f2)
        episode_states.append(state)
        episode_agent_pos.append(info['agent_position'])

        steps += 1
        state, reward, done, info = warehouse.step(np.argmax(action))

        if steps >= max_steps:
            break

        if done: # only steps from solved interations are collected
            state_list.extend(episode_states)
            agent_pos_list.extend(episode_agent_pos)
            f1_list.extend(episode_f1)
            f2_list.extend(episode_f2)
            action_list.extend(episode_actions)

actions = np.array(action_list)[:total_points]
f1 = np.array(f1_list)[:total_points]
f2 = np.array(f2_list)[:total_points]
agent_pos = np.array(agent_pos_list)[:total_points]

# print(actions.shape)
print('f1:', f1.shape)
print('f2:', f2.shape)
print('actions:', actions.shape)
print('agent_pos:', agent_pos.shape)

actions = [np.argmax(row) for row in actions]
labels = [action_labels[a] for a in actions]

# print(actions)
# print(lstms)

# reduce dimensionality to 2D
# print('calculating PCA for f1...')
# pca = PCA(n_components=32)
# f1_reduced = pca.fit_transform(f1)
# print('f1:', f1_reduced.shape)

# print('calculating PCA for f2...')
# pca = PCA(n_components=16)
# f2_reduced = pca.fit_transform(f2)
# print('f2:', f2_reduced.shape)

# print('calculating TSNE for f1...')
# tsne = TSNE(n_components=2, perplexity=30)
# f1_reduced = tsne.fit_transform(f1_reduced)
# print('f1:', f1_reduced.shape)

print('calculating TSNE for f2...')
tsne = TSNE(n_components=2)
f2_reduced = tsne.fit_transform(f2)
print('f2:', f2_reduced.shape)

legend_labels = [action_labels[a] for a in range(4)]


# annotate the env state as dot style (round, triangle, box...) -> use + - |
# for the different relevant states: in horizontal aisle, in vertical aisle, in corner or similar

def get_agent_position_marker(pos):
    if pos[1] == 0 or pos[1] == 6:
        if pos[0] % 3 == 0:
            return '+'
        else:
            return '_'
    if pos[1] > 0 and pos[1] < 6:
        return '|'
    else:
        return ''


# still annotate examples with annotations and show them rendered below

cmap = 'rainbow'
# used to compare both layers if needed
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig, ax2 = plt.subplots(1, figsize=(8, 4))

# used to compare both layers if needed
# scatter = ax1.scatter(f1_reduced[:, 0], f1_reduced[:, 1], c=actions, cmap=cmap, s=70)
# for p, ap in zip(f1_reduced, agent_pos):
#     ax1.annotate(get_agent_position_marker(ap), p)

# handles, _ = scatter.legend_elements()
# legend = ax1.legend(handles, legend_labels, title='Actions')
# ax1.add_artist(legend)

scatter = ax2.scatter(f2_reduced[:, 0], f2_reduced[:, 1], c=actions, cmap=cmap, s=40)
for p, ap in zip(f2_reduced, agent_pos):
    ax2.annotate(get_agent_position_marker(ap), p)
handles, _ = scatter.legend_elements()
legend = ax2.legend(handles, legend_labels, title='Aktion')
ax2.add_artist(legend)

# ax1.tick_params(axis='both', which='both', bottom=False, top=False,
#                 labelbottom=False, right=False, left=False, labelleft=False)
ax2.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)

# used to compare both layers if needed
# fig.suptitle('Outputs of hidden layer with action labels')
# ax1.set_title('Layer 1')
# ax2.set_title('Layer 2')
plt.tight_layout()
plt.show()
