from src.model import DQN, TinyDQN, SmallDQN, BigDQN, BiggerDQN, LstmDQN, BigLstmDQN
from src.agent import QAgent, EntropyQAgent, RecurrentQAgent
from src.warehouse.environment.warehouse_v7 import WarehouseV7

import torch
import numpy as np
from multiprocessing import Pool
import time
from tqdm import tqdm

checkpoint = torch.load('../../models/pytorch_warehouse_v7_xl_12_lstmdqn_09.10_15.57_200000.pt',
                        map_location=torch.device('cpu'))
warehouse = WarehouseV7(num_aisles=checkpoint['config']['num_aisles'], rack_height=checkpoint['config']['rack_height'],
                        min_packets=checkpoint['config']['min_packets'],
                        max_packets=checkpoint['config']['max_packets'], seed=checkpoint['config']['np_seed'])

checkpoint['config']['init_weights'] = False
checkpoint['config']['device'] = 'cpu'
checkpoint['config']['combined_memory'] = False
checkpoint['config']['bidirectional'] = False
checkpoint['config']['ylim'] = (-250, 250)

agent = RecurrentQAgent(env=warehouse, model=LstmDQN, config=checkpoint['config'])
agent.compile()

agent.load_checkpoint(checkpoint)
agent.plot_durations()

# for _ in range(10):
#     agent.play(verbose=True, sleep=0.1, max_steps=175)


def step(i):
    return agent.evaluate_step(index=i, max_steps=350)


if __name__ == '__main__':
    start = time.time()
    pool = Pool(processes=8)
    num_episodes = 10000
    data = []
    for d in tqdm(pool.imap_unordered(step, range(num_episodes)), total=num_episodes):
        data.append(d)
    pool.close()

    done_acc = 0
    step_list = []
    reward_list = []
    print(f'evaluating for {num_episodes} episodes:')
    for index, done, steps, reward in data:
        if done:
            done_acc += 1
            step_list.append(steps)
            reward_list.append(reward)
    print(f'{done_acc * 100 / num_episodes:.2f}% of environments solved!')
    if done_acc > 0:
        print(
            f'with an average of {np.mean(step_list):.2f} steps and an average reward of {np.mean(reward_list):.2f}.')
    end = time.time()
    print(f'evaluation took {end - start:.3f} seconds.')
