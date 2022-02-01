from src.memory import Transition, ReplayMemory, EntropyReplayMemory, CombinedReplayMemory
import numpy as np
import time
from datetime import datetime
import os
import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from abc import ABC, abstractmethod

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display
    from tqdm.notebook import trange
else:
    from tqdm import trange


# clear output for rendering
def clear():
    os.system('clear')


# the code of the agent and the training heavily lean on https://github.com/gandroz/rl-taxi/tree/main/pytorch
class Agent(ABC):
    def __init__(self, *, env, model, config):
        self.config = config
        print(config['tag'])

        # environment and model
        self.env = env
        self.model_class = model
        self.device = self.config['device']

        # seed everything including cuda if GPU is used to train
        self.seed = self.config['np_seed']
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.config['combined_memory']:
            self.memory = CombinedReplayMemory(capacity=self.config['memory_capacity'], seed=self.seed)
        else:
            self.memory = ReplayMemory(capacity=self.config['memory_capacity'], seed=self.seed)
        self.eps_policy = self.config['eps_policy']
        self.loss = self.config['loss']
        self.optimizer = None
        self.policy_model = None
        self.target_model = None
        self.lr_scheduler = None

        # parameters
        self.episode_durations = []
        self.reward_in_episodes = []
        self.epsilon_vec = []

        # values
        self.last_step = 0
        self.last_episode = 0
        self.id = datetime.now().strftime('%d.%m_%H.%M')

    # save necessary information in checkpoint
    def save(self):
        if is_notebook:
            path = f'./models/pytorch_{self.config["tag"]}_{self.id}_{self.last_episode + 1}.pt'
        else:
            path = f'../../models/pytorch_{self.config["tag"]}_{self.id}_{self.last_episode + 1}.pt'
        torch.save({
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_in_episode': self.reward_in_episodes,
            'episode_durations': self.episode_durations,
            'epsilon_vec': self.epsilon_vec,
            'config': self.config
        }, path)

    # moving average used to flatten the training curves
    def _moving_average(self, values, periods=5):
        if len(values) < periods:
            return values

        accumulated_values = np.cumsum(np.insert(values, 0, 0))
        result = (accumulated_values[periods:] - accumulated_values[:-periods]) / periods
        return np.hstack([values[:periods - 1], result])

    # plots steps, rewards and epsilon in the same plot including flattened curves for steps and rewards
    def plot_durations(self):
        lines = []
        fig = plt.figure(1, figsize=(15, 10))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title(f'Training {self.config["tag"]}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        ax1.set_ylim(self.config['ylim'])
        ax1.plot(self.episode_durations, color='C1', alpha=0.2)
        ax1.plot(self.reward_in_episodes, color='C2', alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=300)
        mean_reward = self._moving_average(self.reward_in_episodes, periods=300)
        lines.append(ax1.plot(mean_steps, label='steps', color='C1')[0])
        lines.append(ax1.plot(mean_reward, label='rewards', color='C2')[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilon_vec, label='epsilon', color='C3')[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        if is_notebook:
            display.clear_output(wait=True)
        else:
            plt.show()

        plt.pause(0.001)

    # re-instantiate everything from saved checkpoint
    def load_checkpoint(self, checkpoint):
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_durations = checkpoint['episode_durations']
        self.reward_in_episodes = checkpoint['reward_in_episode']
        self.epsilon_vec = checkpoint['epsilon_vec']
        self.last_episode = len(self.episode_durations)

    # def load_model_from_checkpoint(self, checkpoint):
    #     self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @abstractmethod
    def compile(self):
        ...

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def resume_fit(self, episodes, epsilon):
        ...

    @abstractmethod
    def play(self, verbose, sleep, max_steps):
        ...


class QAgent(Agent):
    def __init__(self, *, env, model, config):
        super().__init__(env=env, model=model, config=config)

    # setup models, optimizer and scheduler
    def compile(self):
        self.policy_model = self.model_class(input_size=self.env.observation_space.n,
                                             output_size=self.env.action_space.n).to(self.device)

        self.target_model = self.model_class(input_size=self.env.observation_space.n,
                                             output_size=self.env.action_space.n).to(self.device)
        # use cloned weights to initialize target model
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        # initialize optimizer and scheduler
        self.optimizer = self.config['optimizer'](params=self.policy_model.parameters(),
                                                  lr=self.config['learning_rate'])
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                           milestones=self.config['scheduler_milestones'],
                                                           gamma=self.config['scheduler_decay'])

    # train the agent
    def fit(self):
        self.episode_durations = []
        self.reward_in_episodes = []
        self.epsilon_vec = []
        self.memory.clear()
        epsilon = 1

        # visualize training progress
        progress_bar = trange(0, self.config['num_episodes'], initial=self.last_episode,
                              total=self.config['num_episodes'])

        for i_episode in progress_bar:
            # initialize
            reward_in_episode = 0
            state = self.env.reset()
            # get epsilon after warum-up is over
            if i_episode >= self.config['warmup_episodes']:
                epsilon = self.eps_policy.get_exploration_rate(i_episode)

            for step in count():
                # take action
                action = self._choose_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                # store every transition in memory
                self._remember(state, action, next_state, reward, done)

                # after warm-up is done: perform weight update step after each
                # step of the agent
                if i_episode >= self.config['warmup_episodes']:
                    self._train_policy_model()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                # check if maximum steps are reached or terminal signal is received
                done = (step == self.config['max_steps_per_episode'] - 1) or done

                # transition to the next state and accumulate rewards for tracking the training
                state = next_state
                reward_in_episode += reward

                # if episode has ended: save the metrics and update progress in progress_bar
                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episodes.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    n = min(self.config['warmup_episodes'], len(self.episode_durations))
                    progress_bar.set_postfix({
                        'reward': np.mean(self.reward_in_episodes[-n:]),
                        'steps': np.mean(self.episode_durations[-n:]),
                        'epsilon': epsilon
                    })
                    # update plot if needed
                    if self.config['plot_progress']:
                        if is_notebook and i_episode % 10 == 0:
                            self.plot_durations()
                        elif i_episode % 100 == 0:
                            self.plot_durations()
                    # break the loop to start new episode
                    break

            # update weights of the target model
            if i_episode % self.config['target_update'] == 0:
                self._update_target_model()

            # save checkpoints during training
            if i_episode % self.config['save_freq'] == 0 and i_episode > 0 and self.config['save_model']:
                self.save()

            self.last_episode = i_episode

    # similar to fit to resume training from a saved checkpoint
    # checkpoint has to be loaded with "load_checkpoint"
    def resume_fit(self, num_episodes, epsilon=None, lr=0.001):
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        if epsilon is None:
            epsilon = self.config['eps_end']
        self.memory.clear()

        progress_bar = trange(0, num_episodes)

        for i_episode in progress_bar:
            reward_in_episode = 0
            state = self.env.reset()

            for step in count():
                action = self._choose_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                self._remember(state, action, next_state, reward, done)

                self._train_policy_model()
                done = (step == self.config['max_steps_per_episode'] - 1) or done

                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episodes.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    n = min(self.config['warmup_episodes'], len(self.episode_durations))
                    progress_bar.set_postfix({
                        'reward': np.mean(self.reward_in_episodes[-n:]),
                        'steps': np.mean(self.episode_durations[-n:]),
                        'epsilon': epsilon
                    })
                    if is_notebook:
                        self.plot_durations()
                    elif i_episode % 100 == 0:
                        self.plot_durations()
                    break

            if i_episode % self.config['target_update'] == 0:
                self._update_target_model()

            if i_episode % self.config['save_freq'] == 0 and i_episode > 0 and self.config['save_model']:
                self.save()

            self.last_episode = len(self.episode_durations)

    # performs one weight update on the policy network
    def _train_policy_model(self):
        # wait until sufficient experiences have been made
        if len(self.memory) < self.config['batch_size']:
            return

        # sample experiences from the memory based on random policy
        transitions = self.memory.sample(self.config['batch_size'])
        # the experiences ( list of transitions: (s, a, ns, r, d )... ) have to be transformed into lists
        # of ((s1, s2, s2) (a1, a2, a3)...) for the network inputs
        batch = Transition(*zip(*transitions))

        # separate each type from the transitions
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # predict q values based on states and collect values for the matching actions that were actually taken
        predicted_q_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))

        # calculate q values for the next state
        next_state_values = self.target_model(next_state_batch).max(1)[0]
        # calculate expected q values using bellman equation
        # ~done_batch evaluates to 0 if done_batch equals true, 1 else
        expected_q_values = reward_batch + (~done_batch * next_state_values * self.config['gamma'])

        # perform optimization step
        # by calculating loss
        loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1).detach())

        # zero out previous gradients
        self.optimizer.zero_grad()
        # backpropagate the loss
        loss.backward()
        # clip gradients
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        # perform optimization step using the chosen optimizer algorithm
        self.optimizer.step()

    # stores a transition into the memory while preserving device and datatype
    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(Transition(torch.tensor([state], device=self.device, dtype=self.config['dtype']),
                                    torch.tensor([action], device=self.device),
                                    torch.tensor([next_state], device=self.device, dtype=self.config['dtype']),
                                    torch.tensor([reward], device=self.device),
                                    torch.tensor([done], device=self.device)))

    # employ epsilon strategy to choose action
    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = self._get_action_for_state(state=state)
        return action

    # run policy network to get prediction
    def _get_action_for_state(self, state, include_q=False):
        with torch.no_grad():
            logits = self.policy_model(torch.tensor([state], device=self.device, dtype=self.config['dtype']))
            # employ max a of Q(s,a) policy and return action index
            action = logits.max(1)[1].item()

        # include q values for visualization if needed
        if include_q:
            return action, logits
        else:
            return action

    # simply re-clone the weights into the target model to keep it up to date
    def _update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    # visualize the actions and observations after each step for humans
    # use "include_q_values=True" to see the predicted q-values under the observations in the rendering
    # set sleep to a higher value to have more time to follow the action and especially the q-values
    def play(self, verbose=True, sleep=0.2, max_steps=100, include_q_values=False):
        done = False
        iteration = 0
        total_reward = 0

        try:
            action_labels = self.env.get_action_labels()

            # initialize everythin
            state = self.env.reset()
            if not is_notebook and verbose:
                clear()
            if verbose:
                self.env.render()
                print(f'Iter: {iteration} - Action: *** - Reward ***')
            time.sleep(sleep)

            # perform and render steps until maximum steps are reached or env is solved
            # always use network to predict
            while not done:
                if not is_notebook and verbose:
                    clear()
                if include_q_values:
                    action, logits = self._get_action_for_state(state, include_q_values)
                    logits = logits.numpy()[0]
                else:
                    action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                if is_notebook:
                    display.clear_output(wait=True)
                if verbose:
                    if include_q_values:
                        self.env.render(current_q_values=logits)
                    else:
                        self.env.render()
                    print(f'Step: {iteration} - Action: {action}({action_labels[action]}) - Reward {reward}')
                time.sleep(sleep)
                if iteration == max_steps:
                    if verbose:
                        print('max steps reached without solution')
                    break
        except KeyboardInterrupt:
            pass

        return done, iteration, total_reward

    # run x episodes and track metrics to evaluate performance
    # basically a variation of the "play" function
    def evaluate(self, *, verbose=False, sleep=0.0, num_episodes=1000, max_steps=200):
        done_acc = 0
        step_list = []
        reward_list = []
        print(f'evaluating for {num_episodes} episodes:')
        for _ in trange(num_episodes):
            done, steps, reward = self.play(verbose=verbose, sleep=sleep, max_steps=max_steps)
            if done:
                done_acc += 1
                step_list.append(steps)
                reward_list.append(reward)
        print(f'{done_acc * 100 / num_episodes:.2f}% of environments solved!')
        if done_acc > 0:
            print(
                f'with an average of {np.mean(step_list):.2f} steps and an average reward of {np.mean(reward_list):.2f}.')
        return done_acc / num_episodes, np.mean(reward_list), np.mean(step_list)

    # used for parallel evaluation
    # the evaluation can be heavily parallelized since every episode is independently calculated
    # makes most sense for the slower recurrent network, the forward pass of the DQN is quite fast already
    def evaluate_step(self, index, verbose=False, sleep=0.0, max_steps=200):
        done, steps, reward = self.play(verbose=verbose, sleep=sleep, max_steps=max_steps)
        return index, done, steps, reward


# the same agent modifierd to capture k observations and make combined predictions using a recurrent network
class RecurrentQAgent(QAgent):
    def __init__(self, *, env, model, config):
        super().__init__(env=env, model=model, config=config)
        self.running_k = self.config['running_k']
        self.running_queue = deque([], maxlen=self.running_k)

    def compile(self):
        self.policy_model = self.model_class(input_size=self.env.observation_space.n,
                                             bidirectional=self.config['bidirectional'],
                                             output_size=self.env.action_space.n).to(self.device)

        self.target_model = self.model_class(input_size=self.env.observation_space.n,
                                             bidirectional=self.config['bidirectional'],
                                             output_size=self.env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        self.optimizer = self.config['optimizer'](params=self.policy_model.parameters(),
                                                  lr=self.config['learning_rate'])
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                           milestones=self.config['scheduler_milestones'],
                                                           gamma=self.config['scheduler_decay'])

    def fit(self):
        self.episode_durations = []
        self.reward_in_episodes = []
        self.epsilon_vec = []
        self.running_queue.clear()
        self.memory.clear()
        epsilon = 1

        progress_bar = trange(0, self.config['num_episodes'], initial=self.last_episode,
                              total=self.config['num_episodes'])

        for i_episode in progress_bar:
            reward_in_episode = 0
            state = self.env.reset()
            for _ in range(self.running_k - 1):
                self.running_queue.append(torch.tensor(np.zeros_like(state)).float().to(self.device))
            self.running_queue.append(torch.tensor(state).float().to(self.device))
            if i_episode >= self.config['warmup_episodes']:
                epsilon = self.eps_policy.get_exploration_rate(i_episode)

            for step in count():
                state_series = torch.stack(list(self.running_queue)).unsqueeze(0)
                action = self._choose_action(state_series, epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.running_queue.append(torch.tensor(next_state).float().to(self.device))
                next_state_series = torch.stack(list(self.running_queue)).unsqueeze(0)

                self._remember_recurrent_states(state_series, action, next_state_series, reward, done)

                if i_episode >= self.config['warmup_episodes']:
                    self._train_policy_model()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                done = (step == self.config['max_steps_per_episode'] - 1) or done
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episodes.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    n = min(self.config['warmup_episodes'], len(self.episode_durations))
                    progress_bar.set_postfix({
                        'reward': np.mean(self.reward_in_episodes[-n:]),
                        'steps': np.mean(self.episode_durations[-n:]),
                        'epsilon': epsilon
                    })
                    if self.config['plot_progress']:
                        if is_notebook:
                            self.plot_durations()
                        elif i_episode % 100 == 0:
                            self.plot_durations()
                    break

            if i_episode % self.config['target_update'] == 0:
                self._update_target_model()

            if i_episode % self.config['save_freq'] == 0 and i_episode > 0 and self.config['save_model']:
                self.save()

            self.last_episode = i_episode

    def resume_fit(self, num_episodes, epsilon=None, lr=0.001):
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        if epsilon is None:
            epsilon = self.config['eps_end']
        self.memory.clear()
        self.running_queue.clear()

        progress_bar = trange(0, num_episodes)

        for i_episode in progress_bar:
            reward_in_episode = 0
            state = self.env.reset()
            for _ in range(self.running_k - 1):
                self.running_queue.append(torch.tensor(np.zeros_like(state)).float().to(self.device))
            self.running_queue.append(torch.tensor(state).float().to(self.device))

            for step in count():
                state_series = torch.stack(list(self.running_queue)).unsqueeze(0)
                action = self._choose_action(state_series, epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.running_queue.append(torch.tensor(next_state).float().to(self.device))
                next_state_series = torch.stack(list(self.running_queue)).unsqueeze(0)

                self._remember_recurrent_states(state_series, action, next_state_series, reward, done)

                self._train_policy_model()
                done = (step == self.config['max_steps_per_episode'] - 1) or done

                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episodes.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    n = min(self.config['warmup_episodes'], len(self.episode_durations))
                    progress_bar.set_postfix({
                        'reward': np.mean(self.reward_in_episodes[-n:]),
                        'steps': np.mean(self.episode_durations[-n:]),
                        'epsilon': epsilon
                    })
                    if is_notebook:
                        self.plot_durations()
                    elif i_episode % 100 == 0:
                        self.plot_durations()
                    break

            if i_episode % self.config['target_update'] == 0:
                self._update_target_model()

            if i_episode % self.config['save_freq'] == 0 and i_episode > 0 and self.config['save_model']:
                self.save()

            self.last_episode = len(self.episode_durations)

    def _remember_recurrent_states(self, state, action, next_state, reward, done):
        self.memory.push(Transition(state.to(self.device),
                                    torch.tensor([action], device=self.device),
                                    next_state.to(self.device),
                                    torch.tensor([reward], device=self.device),
                                    torch.tensor([done], device=self.device)))

    def _choose_action(self, state_series, epsilon):
        if self.rng.uniform() < epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = self._get_action_for_state(state_series=state_series)
        return action

    def _get_action_for_state(self, state_series, include_q=False):
        with torch.no_grad():
            logits = self.policy_model(state_series)
            action = logits.max(1)[1].item()
        if include_q:
            return action, logits
        else:
            return action

    def play(self, verbose=True, sleep=0.2, max_steps=100, include_q_values=False):
        done = False
        iteration = 0
        total_reward = 0

        try:
            action_labels = self.env.get_action_labels()

            state = self.env.reset()
            if not is_notebook and verbose:
                clear()
            if verbose:
                self.env.render()
                print(f'Step: {iteration} - Action: *** - Reward ***')
            time.sleep(sleep)

            self.running_queue.clear()
            for _ in range(self.running_k - 1):
                self.running_queue.append(torch.tensor(np.zeros_like(state)).float().to(self.device))
            self.running_queue.append(torch.tensor(state).float().to(self.device))

            while not done:
                if not is_notebook and verbose:
                    clear()
                state_series = torch.stack(list(self.running_queue)).unsqueeze(0)
                if include_q_values:
                    action, logits = self._get_action_for_state(state_series, include_q_values)
                    logits = logits.numpy()[0]
                else:
                    action = self._get_action_for_state(state_series)

                iteration += 1
                state, reward, done, info = self.env.step(action)
                self.running_queue.append(torch.tensor(state).float().to(self.device))
                total_reward += reward

                if is_notebook:
                    display.clear_output(wait=True)
                if verbose:
                    if include_q_values:
                        self.env.render(current_q_values=logits)
                    else:
                        self.env.render()
                    print(f'Step: {iteration} - Action: {action}({action_labels[action]}) - Reward {reward}')
                time.sleep(sleep)
                if iteration == max_steps:
                    if verbose:
                        print('max steps reached without solution')
                    break
        except KeyboardInterrupt:
            pass

        return done, iteration, total_reward


# an idea to incentivize the exploration of the env -> work in progress / future work
class EntropyQAgent(QAgent):
    def __init__(self, *, env, model, config):
        super().__init__(env=env, model=model, config=config)
        self.entropy_episodes = config['entropy_episodes']
        self.entropy_memory = EntropyReplayMemory(capacity=self.config['entropy_memory_capacity'])

    def fit(self):
        self.episode_durations = []
        self.reward_in_episodes = []
        self.epsilon_vec = []
        self.memory.clear()
        self.entropy_memory.clear()
        epsilon = 1

        progress_bar = trange(0, self.config['num_episodes'], initial=self.last_episode,
                              total=self.config['num_episodes'])

        for i_episode in progress_bar:
            reward_in_episode = 0
            steps_in_episode = set()
            state = self.env.reset()
            if i_episode >= self.config['warmup_episodes']:
                epsilon = self.eps_policy.get_exploration_rate(i_episode)

            for step in count():
                action = self._choose_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                if i_episode < self.entropy_episodes:
                    # newness_score = self.entropy_memory.get_newness_score(state, next_state)
                    # if reward >= -1:
                    #     reward = newness_score
                    # else:
                    #     reward = -1
                    agent_step = next_state[:2].tobytes()
                    if reward >= -1:
                        if agent_step in steps_in_episode:
                            reward = 0
                        else:
                            reward = 1
                    else:
                        reward = -1
                    steps_in_episode.add(agent_step)

                if i_episode < self.entropy_episodes:
                    self._remember_entropy(state, action, next_state, reward, done)
                self._remember(state, action, next_state, reward, done)

                if i_episode >= self.config['warmup_episodes']:
                    self._train_policy_model_with_entropy(i_episode)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                done = (step == self.config['max_steps_per_episode'] - 1) or done
                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episodes.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    n = min(self.config['warmup_episodes'], len(self.episode_durations))
                    progress_bar.set_postfix({
                        'reward': np.mean(self.reward_in_episodes[-n:]),
                        'steps': np.mean(self.episode_durations[-n:]),
                        'epsilon': epsilon
                    })
                    if is_notebook:
                        self.plot_durations()
                    elif i_episode % 100 == 0:
                        self.plot_durations()
                    break

            if i_episode % self.config['target_update'] == 0:
                self._update_target_model()

            if i_episode % self.config['save_freq'] == 0 and i_episode > 0 and self.config['save_model']:
                self.save()

            self.last_episode = i_episode

    def _remember_entropy(self, state, action, next_state, reward, done):
        self.entropy_memory.push(Transition(torch.tensor([state], device=self.device, dtype=self.config['dtype']),
                                            torch.tensor([action], device=self.device),
                                            torch.tensor([next_state], device=self.device, dtype=self.config['dtype']),
                                            torch.tensor([reward], device=self.device),
                                            torch.tensor([done], device=self.device)))

    def _train_policy_model_with_entropy(self, episode):
        if len(self.entropy_memory) < self.config['batch_size']:
            return

        if episode < self.entropy_episodes:
            transitions = self.entropy_memory.sample(self.config['batch_size'])
        else:
            transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        predicted_q_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (~done_batch * next_state_values * self.config['gamma'])

        loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        if self.config['clip_gradient']:
            for param in self.policy_model.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
