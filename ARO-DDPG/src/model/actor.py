import torch
from torch import nn
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(
        self,
        env,
        act_func=nn.ReLU,
        lr=3e-4,
        state_dim=None,
        action_dim=None,
        n_hidden_nodes=256,
        device=torch.device("cpu"),
    ):
        super(Actor, self).__init__()

        self.env = env

        self.device = device

        if state_dim is None:
            self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = state_dim

        if action_dim is None:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, n_hidden_nodes),
            act_func(),
            nn.Linear(n_hidden_nodes, n_hidden_nodes),
            act_func(),
            nn.Linear(n_hidden_nodes, self.action_dim),
        )

        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        state = state.to(self.device)
        return self.policy_net(state).to("cpu")

    def get_action(self, state):
        action = self(state)
        action = torch.tanh(action)

        return action