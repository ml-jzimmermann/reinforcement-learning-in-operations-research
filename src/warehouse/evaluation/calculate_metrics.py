import numpy as np


big = {"warehouse_v7_thesis_big_dqn_7": {"solved": 0.06283, "reward": 108.60958141015439, "steps": 68.25752029285373}, "warehouse_v7_thesis_big_dqn_3": {"solved": 0.79639, "reward": 104.90140509047075, "steps": 72.68265548286644}, "warehouse_v7_thesis_big_dqn_5": {"solved": 0.00352, "reward": 125.78125, "steps": 51.75568181818182}, "warehouse_v7_thesis_big_dqn_11": {"solved": 0.0, "reward": 0.0, "steps": 0.0}, "warehouse_v7_thesis_big_dqn_13": {"solved": 0.14572, "reward": 111.33763381828163, "steps": 65.99972550096075}}
big_vp = {"warehouse_v7_thesis_big_vp_dqn_5": {"solved": 0.36396, "reward": 94.86611166062205, "steps": 57.10374766457853}, "warehouse_v7_thesis_big_vp_dqn_13": {"solved": 0.0, "reward": 0, "steps": 0}, "warehouse_v7_thesis_big_vp_dqn_3": {"solved": 0.4752, "reward": 93.17274831649831, "steps": 59.281734006734006}, "warehouse_v7_thesis_big_vp_dqn_7": {"solved": 0.16154, "reward": 91.3706821839792, "steps": 59.86665841277702}, "warehouse_v7_thesis_big_vp_dqn_11": {"solved": 0.17742, "reward": 94.62394318566115, "steps": 54.63499041821666}}

avg_solved = 0
avg_reward = 0
avg_steps = 0

max_solved = 0
max_reward = 0
min_steps = 1000

for k, v in big.items():
    avg_solved += v['solved']
    avg_reward += v['reward']
    avg_steps += v['steps']

    if v['solved'] > max_solved:
        max_solved = v['solved']

    if v['reward'] > max_reward:
        max_reward = v['reward']

    if v['steps'] < min_steps and v['steps'] > 0:
        min_steps = v['steps']

print(f'steps: {avg_steps / 4:.2f} - min: {min_steps:.2f}')
print(f'reward: {avg_reward / 4:.2f} - max: {max_reward:.2f}')
print(f'solved: {avg_solved * 100 / 4:.2f} - max: {max_solved * 100:.2f}')

