import numpy as np

import torch
from torch.autograd import Variable
from utils import *


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss

def update_params(batch, targ_adv_fun):
    vloss = []
    ploss = []
    rewards = torch.tensor(np.array(batch.reward))
    masks = torch.tensor(np.array(batch.mask))
    actions = torch.Tensor(np.concatenate(batch.action, 0)).detach()
    states = torch.tensor(np.array(batch.state)).detach()
    values = val(states)

    targets, advantages = targ_adv_fun(rewards, masks, actions, values)

    targets = targets.detach()
    advantages = advantages.detach()

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(val, torch.Tensor(flat_params))
        for param in val.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        value_loss = (val(states) - targets).pow(2).mean()
        vloss.append(value_loss)

        # weight decay
        for param in val.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(val).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(val).double().numpy(), maxiter=40)
    set_flat_params_to(val, torch.Tensor(flat_params))

    action_means, action_log_stds, action_stds = agent(states)
    fixed_log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds).data.clone().detach()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = agent(states)
        else:
            action_means, action_log_stds, action_stds = agent(states)
                
        log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        action_loss = -(advantages * torch.exp(log_prob - fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = agent(states)

        mean0 = mean1.data.detach()
        log_std0 = log_std1.data.detach()
        std0 = std1.data.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    policy_loss = trpo_step(agent, get_loss, get_kl, max_kl, damping)
    ploss.append(policy_loss)
    return vloss, ploss, advantages

def get_trpo_tar_mean_adv(rewards, masks, actions, values, gamma = 0.99, tau = 0.97):
    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = returns
    advantages = (advantages - advantages.mean()) / advantages.std()

    return targets, advantages

def get_atrpo_tar_adv(rewards, masks, actions, values):
    ro = torch.mean(rewards)

    advantages = torch.Tensor(actions.size(0),1)
    targets = torch.Tensor(actions.size(0),1)

    prev_value = 0
    for i in reversed(range(rewards.size(0))):
        targets[i] = rewards[i] - ro + prev_value * masks[i]
        advantages[i] = rewards[i] - ro + prev_value * masks[i] - values.data[i]

        prev_value = values.data[i, 0]

    return targets, advantages