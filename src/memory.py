from collections import namedtuple, deque, defaultdict
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory:
    def __init__(self, capacity, seed=None):
        self.experiences = deque([], maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(self, transition):
        self.experiences.append(transition)

    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)

    def clear(self):
        self.experiences.clear()

    def __len__(self):
        return len(self.experiences)


# idea from https://arxiv.org/abs/1712.01275
class CombinedReplayMemory(ReplayMemory):
    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.last_experience = None

    def push(self, transition):
        self.last_experience = transition
        self.experiences.append(transition)

    def clear(self):
        self.last_experience = None
        self.experiences.clear()

    def sample(self, batch_size):
        batch = random.sample(self.experiences, batch_size - 1)
        batch.append(self.last_experience)
        return batch


# effort to implement sth inspired by https://arxiv.org/abs/2103.04551 -> future work
class EntropyReplayMemory(ReplayMemory):
    def __init__(self, capacity):
        super(EntropyReplayMemory, self).__init__(capacity=capacity)
        self.transitions = defaultdict(int)

    def push(self, transition):
        self.transitions[(transition.state.numpy()[:2].tobytes(),
                          transition.next_state.numpy()[:2].tobytes())] += 1
        self.experiences.append(transition)

    def get_newness_score(self, state, next_state):
        transition_count = self.transitions[(state[:2].tobytes(), next_state[:2].tobytes())]
        return 5 if transition_count == 0 else 1 / transition_count
