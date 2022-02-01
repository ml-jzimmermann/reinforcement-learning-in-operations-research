import torch.nn as nn
import torch.nn.functional as F

# taken from https://github.com/gandroz/rl-taxi/blob/main/pytorch/model.py
class EmbeddingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbeddingDQN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=5)
        self.l1 = nn.Linear(in_features=5, out_features=50)
        self.l2 = nn.Linear(in_features=50, out_features=50)
        self.out = nn.Linear(in_features=50, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.l1(self.embed(x)))
        x = F.relu(self.l2(x))
        x = self.out(x)
        return x


# simple feed-forward network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(in_features=input_size, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)
        return x


# lstm network to handle multiple observations at the same time
# similar to the atari idea of seeing 4 connnected frames -> future work
class LstmDQN(nn.Module):
    def __init__(self, input_size, output_size, bidirectional=False):
        super(LstmDQN, self).__init__()
        # lstms to process the sequential inputs
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True,
                            bidirectional=bidirectional)
        # make predictions based on hidden representation
        self.out = nn.Linear(in_features=128 if bidirectional else 64, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # only use latest hidden state for prediction
        x = self.out(x)
        return x
