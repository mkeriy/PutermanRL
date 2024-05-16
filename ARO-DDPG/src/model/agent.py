import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random

from src.utils import Logger
from src.model.actor import Actor
from src.model.critic import Critic


class Agent:
    def __init__(self, env, config):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        self.env = env

        self.device = config["device"]

        self.log = config["log"]

        self.logger = Logger()

        self.state_dim = config["state_dim"]

        self.action_dim = config["action_dim"]

        self.actor = Actor(
            env,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=config["lr_actor"],
            n_hidden_nodes=config["actor_hidden"],
            device=self.device,
        )
        self.actor.to(self.device)
        self.actor_target = Actor(
            env,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=config["lr_actor"],
            n_hidden_nodes=config["actor_hidden"],
            device=self.device,
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.to(self.device)

        self.critic = Critic(
            env,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=config["lr_critic"],
            n_hidden_nodes=config["critic_hidden"],
            device=self.device,
        )
        self.critic.to(self.device)
        self.critic_target = Critic(
            env,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=config["lr_critic"],
            n_hidden_nodes=config["critic_hidden"],
            device=self.device,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.to(self.device)

        self.rho = torch.tensor([0.0], requires_grad=True, dtype=torch.float)

        self.rho_optim = Adam([self.rho], lr=config["lr_rho"])

        self.tau = config["tau"]

    def update(
        self,
        buffer,
        batch_size=256,
        gradient_steps=1,
        critic_freq=1,
        actor_freq=2,
        rho_freq=2,
    ):
        for n_updates in range(1, gradient_steps + 1):
            state, reward, action, done, next_state = buffer.sample(batch_size)
            n_updates += 1

            if n_updates % critic_freq == 0:
                with torch.no_grad():
                    next_action = self.actor_target.get_action(next_state)
                    q1, q2 = self.critic_target(
                        torch.cat([next_state, next_action], axis=1)
                    )
                    temp = (1 - done) * (torch.min(q1, q2).reshape(-1))
                    q_target = reward + temp

                q_values = self.critic(torch.cat([state, action], axis=1))
                critic_loss = 0.5 * sum(
                    [
                        F.mse_loss(current_q.reshape(-1) + self.rho, q_target)
                        for current_q in q_values
                    ]
                )

                self.critic.optimizer.zero_grad()
                self.rho_optim.zero_grad()

                critic_loss.backward()

                self.critic.optimizer.step()
                self.rho_optim.step()

                if self.log:
                    self.logger.add_scalar(
                        "critic value max", torch.max(q_values[0]).item()
                    )
                    self.logger.add_scalar(
                        "critic value min", torch.min(q_values[0]).item()
                    )
                    self.logger.add_scalar("critic_loss", critic_loss.item())
                    self.logger.add_scalar(
                        "reward-rho", torch.max(reward - self.rho).item()
                    )
                    self.logger.add_scalar("critic next", torch.max(temp).item())

                    self.logger.add_scalar("rho max", torch.max(self.rho).item())
                    # self.logger.add_scalar("rho min",torch.min(self.rho).item())

            if n_updates % actor_freq == 0:
                action_pi = self.actor.get_action(state)

                q1_pi, q2_pi = self.critic(torch.cat([state, action_pi], axis=1))
                q_pi = torch.min(q1_pi, q2_pi).reshape(-1)
                actor_loss = -(q_pi).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                with torch.no_grad():
                    for params, params_target in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                    ):
                        params_target.data.mul_(self.tau)
                        params_target.data.add_((1 - self.tau) * params)

                    for params, params_target in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        params_target.data.mul_(self.tau)
                        params_target.data.add_((1 - self.tau) * params)

                if self.log:
                    self.logger.add_scalar("actor loss", actor_loss.item())

    def evaluate(self, epi_len=1000, n_iter=1):
        self.actor.train(False)
        self.critic.train(False)

        with torch.no_grad():
            total_reward = 0
            total_steps = 0
            for epi in range(n_iter):
                state = self.env.reset()
                done = False
                for epi_steps in range(1, epi_len + 1):
                    total_steps += 1
                    action = self.actor.get_action(torch.tensor(state).float())
                    next_state, reward, done, _ = self.env.step(action.detach().numpy())
                    total_reward += reward
                    state = next_state

        return (total_reward / n_iter), (total_reward / total_steps)

    def get_action(self, state):
        return self.actor.get_action(state)
