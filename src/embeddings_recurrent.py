from collections import deque
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as T

from warehouse_v7 import WarehouseV7


# the same as embeddings.py for recurrent LstmDQN
# see embeddings.py for further details


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        lstm_out = x[:, -1, :]
        x = self.out(lstm_out)
        return x, lstm_out


checkpoint = T.load('../models/pytorch_warehouse_v7_bigger_lstmdqn_lr2_28.09_07.54_200000.pt',
                    map_location=T.device('cpu'))

warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'])

action_labels = warehouse.get_action_labels()

model = Model(input_size=warehouse.observation_space.n, output_size=warehouse.action_space.n)
model.load_state_dict(checkpoint['policy_model_state_dict'])

running_k = 4
running_queue = deque([], maxlen=running_k)

action_list = []
lstm_list = []
episodes = 15
max_steps = 130

print('generating observations...')
for episode in tqdm(range(episodes)):
    running_queue.clear()
    episode_actions = []
    episode_lstms = []

    state = warehouse.reset()
    for _ in range(running_k - 1):
        running_queue.append(T.tensor(np.zeros_like(state)).float())
    running_queue.append(T.tensor(state).float())

    done = False
    steps = 0

    while not done:
        inputs = T.stack(list(running_queue)).unsqueeze(0)
        action, lstm = model(inputs)
        action, lstm = action.detach().squeeze().numpy(), lstm.detach().squeeze().numpy()
        episode_actions.append(action)
        episode_lstms.append(lstm)
        steps += 1
        state, reward, done, info = warehouse.step(np.argmax(action))
        running_queue.append(T.tensor(state).float())

        if steps >= max_steps:
            for e in episode_actions:
                action_list.append(e)
            for e in episode_lstms:
                lstm_list.append(e)
            break

        if done:
            for e in episode_actions:
                action_list.append(e)
            for e in episode_lstms:
                lstm_list.append(e)

actions = np.array(action_list)
lstms = np.array(lstm_list)

# print(actions.shape)
print(lstms.shape)

actions = [np.argmax(row) for row in actions]
labels = [action_labels[a] for a in actions]

# print(actions)
# print(lstms)

print('calculating PCA...')
pca = PCA(n_components=6)
lstm_reduced = pca.fit_transform(lstms)
print(lstm_reduced.shape)

print('calculating TSNE...')
tsne = TSNE(n_components=2, perplexity=30)
lstm_reduced = tsne.fit_transform(lstm_reduced)

# print(lstm_reduced)
print(lstm_reduced.shape)
# print(labels)

legend_labels = [action_labels[a] for a in range(4)]

# for i in range(lstm_reduced.shape[0]):
fig, ax = plt.subplots()
scatter = ax.scatter(lstm_reduced[:, 0], lstm_reduced[:, 1], c=actions)
handles, _ = scatter.legend_elements()
legend = ax.legend(handles, legend_labels, title='Actions')
ax.add_artist(legend)
# plt.axis('off')
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)
plt.title('Outputs of LSTMs with action labels')
plt.show()
