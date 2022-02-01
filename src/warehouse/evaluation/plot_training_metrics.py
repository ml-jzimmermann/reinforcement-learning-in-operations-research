import torch as T
import os
import matplotlib.pyplot as plt
import json
import numpy as np


def moving_average(values, periods=5):
    if len(values) < periods:
        return values

    accumulated_values = np.cumsum(np.insert(values, 0, 0))
    result = (accumulated_values[periods:] - accumulated_values[:-periods]) / periods
    return np.hstack([values[:periods - 1], result])


def get_seed_from_file(file_name):
    array = file_name.split('_')
    try:
        if 'big_vp_dqn' in file_name:
            return int(array[7])
        else:
            return int(array[7])
    except:
        return 0



colors = {3: 'blue', 5: 'orange', 7: 'green', 11: 'red', 13: 'purple'}

filepath = '../../../models/final'
group = 'big_vp_dqn'

fig = plt.figure(1, figsize=(8, 4))
ax1 = fig.add_subplot(111)
ax1.set_ylim((70, 101))
ax1.set_ylabel('Schritte')
ax1.set_xlabel('Episode')

for file in sorted(os.listdir(filepath), key=lambda x: 0 if 'pytorch' not in x else get_seed_from_file(x)):
    if group in file:
        checkpoint = T.load(f'{filepath}/{file}', map_location=T.device('cpu'))
        steps = checkpoint['episode_durations']
        rewards = checkpoint['reward_in_episode']

        seed = get_seed_from_file(file)

        ax1.plot(range(500, len(steps)), moving_average(steps, 500)[500:],
                 linewidth=0.5, label=f'Seed {seed:3d}', c=colors[seed])

plt.legend()
plt.tight_layout()
plt.show()
