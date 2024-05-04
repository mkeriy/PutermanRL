import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def weights_init_normal(layers, mean, std):
    for layer in layers:
        layer.weight.data.normal_(mean, std)


class FeedForwardNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=32):
        super(FeedForwardNet, self).__init__()
        self.in_dim = in_dim

        self.fc1 = nn.Linear(in_dim, hidden_size)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(hidden_size, out_dim)

    def forward(self, q):
        x = F.relu(self.fc1(q))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FeedForwardNet_v2(nn.Module):
    def __init__(self, in_dim, in2_dim, out_dim, hidden_size=32):
        super(FeedForwardNet_v2, self).__init__()

        self.in_dim = in_dim
        self.in2_dim = in2_dim

        self.fcs1 = nn.Linear(in_dim, hidden_size)
        # self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(hidden_size, hidden_size)
        # self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(in2_dim, hidden_size)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(hidden_size, out_dim)
        # self.fc3.weight.data.uniform_(-0.003,0.003)
        # self.out_activation = nn.Sigmoid()

    def forward(self, state, noise):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(noise))
        x = torch.cat((s2, a1), dim=1)

        # print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x)
        # x = self.out_activation(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # self.l1 = nn.Linear(state_dim + action_dim, 400)
        # self.l2 = nn.Linear(400 , 300)
        # self.l3 = nn.Linear(300, 1)

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

    def forward(self, x, u):
        x = F.tanh(self.l1(torch.cat([x, u], 1)))
        x = F.tanh(self.l2(x))
        x = self.l3(x)
        return x


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Value, self).__init__()

        # self.l1 = nn.Linear(state_dim + action_dim, 400)
        # self.l2 = nn.Linear(400 , 300)
        # self.l3 = nn.Linear(300, 1)

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = self.l3(x)
        return x


class EnvModel(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
        super(EnvModel, self).__init__()
        # build network layers
        self.fc1 = nn.Linear(N_STATES + N_ACTIONS, H1Size)
        self.fc2 = nn.Linear(H1Size, H2Size)
        self.statePrime = nn.Linear(H2Size, N_STATES)
        self.reward = nn.Linear(H2Size, 1)
        self.done = nn.Linear(H2Size, 1)

        # initialize layers
        weights_init_normal(
            [self.fc1, self.fc2, self.statePrime, self.reward, self.done], 0.0, 0.1
        )
        # utils.weights_init_xavier([self.fc1, self.fc2, self.statePrime, self.reward, self.done])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        statePrime_value = self.statePrime(x)
        reward_value = self.reward(x)
        done_value = self.done(x)
        done_value = F.sigmoid(done_value)

        return statePrime_value, reward_value, done_value


class TransitionModel(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
        super(TransitionModel, self).__init__()
        # build network layers
        self.fc1 = nn.Linear(N_STATES + N_ACTIONS, H1Size)
        self.fc2 = nn.Linear(H1Size, H2Size)
        self.statePrime = nn.Linear(H2Size, N_STATES)

        # initialize layers
        # weights_init_normal([self.fc1, self.fc2, self.statePrime], 0.0, 0.1)
        # utils.weights_init_xavier([self.fc1, self.fc2, self.statePrime, self.reward, self.done])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)

        statePrime_value = self.statePrime(x)

        return statePrime_value


class FeedForwardNet_v2_dist(nn.Module):
    def __init__(self, in_dim, in2_dim, out_dim, hidden_size=32):
        super(FeedForwardNet_v2_dist, self).__init__()

        self.in_dim = in_dim
        self.in2_dim = in2_dim

        self.fcs = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.fcq = nn.Sequential(nn.Linear(in2_dim, hidden_size), nn.Tanh())

        self.fcm = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Tanh())

        self.action_mean = nn.Linear(hidden_size, out_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state, noise):

        s = self.fcs(state)
        q = self.fcq(noise)
        x = torch.cat((s, q), dim=1)
        x = self.fcm(x)

        action_mean = self.action_mean(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, x, noise):
        with torch.no_grad():
            # action_mean, _, action_std = self.forward(torch.tensor(x).unsqueeze(0), torch.tensor(noise).unsqueeze(0))
            action_mean, _, action_std = self.forward(
                torch.tensor(x), torch.tensor(noise)
            )
            action = torch.normal(action_mean, action_std)
        return action


class FeedForwardNet_v2_dist_2(nn.Module):
    def __init__(self, in_dim, in2_dim, out_dim, hidden_size=32):
        super(FeedForwardNet_v2_dist, self).__init__()

        self.in_dim = in_dim
        self.in2_dim = in2_dim

        self.fcs = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.fcq = nn.Sequential(nn.Linear(in2_dim, 1), nn.Tanh())

        self.fcm = nn.Sequential(nn.Linear(hidden_size + 1, hidden_size), nn.Tanh())

        self.action_mean = nn.Linear(hidden_size, out_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, state, noise):

        s = self.fcs(state)
        q = self.fcq(noise)
        x = torch.cat((s, q), dim=1)
        x = self.fcm(x)

        action_mean = self.action_mean(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, x, noise):
        with torch.no_grad():
            # action_mean, _, action_std = self.forward(torch.tensor(x).unsqueeze(0), torch.tensor(noise).unsqueeze(0))
            action_mean, _, action_std = self.forward(
                torch.tensor(x), torch.tensor(noise)
            )
            action = torch.normal(action_mean, action_std)
            action = torch.clamp(action, -1, 1)
        return action


class Forvard_mean_std(nn.Module):
    def __init__(self, observ_shape, n_actions, h_size):
        super(Forvard_mean_std, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(observ_shape, h_size),
            nn.Tanh(),
            nn.Linear(h_size, h_size),
            nn.Tanh(),
        )

        self.action_mean = nn.Linear(h_size, n_actions)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, x):
        x = self.linear(x)

        action_mean = self.action_mean(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, x):
        with torch.no_grad():
            action_mean, _, action_std = self.forward(torch.tensor(x).unsqueeze(0))
            action = torch.normal(action_mean, action_std)

            action = torch.clamp(action, -1, 1)

        return action[0]
