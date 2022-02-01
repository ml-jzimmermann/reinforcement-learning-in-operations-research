import numpy as np
from abc import ABC, abstractmethod


# simple implementations of different epsilon strategies
class EpsilonGreedyPolicy(ABC):
    def __init__(self, eps_max, eps_min):
        self.eps_max = eps_max
        self.eps_min = eps_min

    @abstractmethod
    def get_exploration_rate(self, episode):
        ...


class LinearEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *, eps_max, eps_min, warm_up_episodes=100, decay_episodes=80000):
        super(LinearEpsilonGreedyPolicy, self).__init__(eps_max, eps_min)
        self.warm_up_episodes = warm_up_episodes
        self.decay_episodes = decay_episodes

    def get_exploration_rate(self, episode):
        if episode < self.warm_up_episodes:
            return self.eps_max
        return self.eps_min + np.maximum(0, (1 - self.eps_min) - (1 - self.eps_min) / self.decay_episodes * (
                episode - self.warm_up_episodes))


class ExponentialEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *, eps_max, eps_min, decay):
        super(ExponentialEpsilonGreedyPolicy, self).__init__(eps_max, eps_min)
        self.decay = decay

    def get_exploration_rate(self, episode):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(-1. * episode / self.decay)
