from dataclasses import dataclass
import typing as tt
import numpy as np
import torch
import torch.nn as nn

import ptan.ignite as ptan_ignite
from ptan.experience import ExperienceFirstLast, \
    ExperienceSourceFirstLast, PrioritizedReplayBuffer

from ignite.contrib.handlers import tensorboard_logger as tb_logger


class HyperParameters(dataclass):
    env_name: str
    run_name: str

    buff_size: int
    buff_initial_size: int
    target_net_sync: int

    stop_reward: float
    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99

GAME_PARAMS = {
    'invaders': HyperParameters(
        env_name="SpaceInvadersNoFrameskip-v4",
        stop_reward=500.0,
        run_name='invaders',
        replay_size=10_000_000,
        replay_initial=50_000,
        target_net_sync=10_000,
        epsilon_frames=10_000_000,
        learning_rate=0.00025,
    ),
}

# gen batch 

def gen_batch(buffer:PrioritizedReplayBuffer,params:HyperParameters):
    buffer.populate(params.replay_initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(params.batch_size)


# functions : unpack_batch , calc_loss , attach_ignite


def unpack_batch(batch:tt.List[ExperienceFirstLast]):
    states , actions , rewards , dones , last_states =[],[],[],[],[]
    #  return states , actions , rewards , dones ,last_states : np.ndarray
    for transition in batch:
        states.append(transition.state)
        actions.append(transition.action)
        rewards.append(transition.reward)
        dones.append(transition.last_state is None)
        if transition.last_state is None:
            last_state = transition.state
        else:
            last_state = transition.last_state
        last_states.append(last_state)
    return np.array(states, copy=False), \
        np.array(actions), \
        np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=bool), \
        np.array(last_states, copy=False)


def calc_loss(tgt_net:nn.Module,net:nn.Module,
              batch:tt.List[ExperienceFirstLast],
              batch_weights: np.ndarray,
              gamma: float,
              device: torch.device='cuda')-> tt.Tuple[torch.Tensor, np.ndarray]:
    # return overall loss + individual loss for each transition to update the prioriy                            
    states , actions , rewards , dones , last_states =unpack_batch(batch)   

    states_t= torch.as_tensor(states).to(device)
    actions_t= torch.tensor(actions).to(device)
    rewards_t= torch.tensor(rewards).to(device)
    dones_t= torch.BoolTensor(dones).to(device)
    last_states_t= torch.as_tensor(last_states).to(device)

    batch_weights_t = torch.tensor(batch_weights)

    all_q_vals:torch.Tensor = net(states_t)
    predicted_q_vals = all_q_vals.gather(
        dim=1,index=actions_t.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        all_target_q_vals:torch.Tensor = tgt_net(last_states_t)
        target_q_vals,_=all_target_q_vals.max(dim=1)
        target_q_vals[dones_t]=0.0
        target_q_vals = gamma*target_q_vals.detach() + rewards_t

    losses = (target_q_vals - predicted_q_vals)**2
    # to correct the bias for sampling
    losses = losses*batch_weights_t
    loss = losses.mean()
    return loss ,(losses + 1e-5).data.cpu().numpy() # + small number to avoid priors=0


    