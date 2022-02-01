import torch as T
import matplotlib.pyplot as plt


filepath = '../../../models/final'

fig = plt.figure(1, figsize=(8, 4))
ax1 = fig.add_subplot(111)
ax1.set_ylim((-0.1, 1.1))
ax1.set_ylabel('Epsilon')
ax1.set_xlabel('Episode')

checkpoint = T.load(f'{filepath}/pytorch_warehouse_v7_thesis_big_dqn_3_18.12_17.53_50000.pt', map_location=T.device('cpu'))
eps = checkpoint['epsilon_vec']

ax1.plot(range(0, len(eps)), eps, c='firebrick', linewidth=1)
# ax1.plot(range(0, len(eps)), [0.02 for _ in range(0, len(eps))], c='black', linewidth=0.5)
plt.tight_layout()
plt.show()
