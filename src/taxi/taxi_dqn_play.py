from src.model import EmbeddingDQN
from src.agent import QAgent
import gym
import torch

# run and observe the agent in action in the taxi environment
# to enable the output terminal to replace the last observation in place: enable "emulate terminal in ouput console"
taxi = gym.make('Taxi-v3').env

obs, reward, done, info = taxi.step(1)
# taxi.render()
# print(obs, reward, done, info)
# exit()

checkpoint = torch.load('../../models/final/pytorch_17.08_08.44_14000.pt')

agent = QAgent(env=taxi, model=EmbeddingDQN, config=checkpoint['config'])
agent.compile()

agent.load_checkpoint(checkpoint)

for _ in range(5):
    agent.play(verbose=True, sleep=0.4, max_steps=25)
