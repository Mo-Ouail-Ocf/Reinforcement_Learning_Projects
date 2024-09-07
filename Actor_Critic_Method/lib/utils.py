import torch.nn as nn
import torch
import numpy as np
from .config import Hyperparams
from ptan.experience import ExperienceFirstLast  
import time
import sys

# batch : list[ExperienceFirstLast]

def unpack_batch(batch:list[ExperienceFirstLast],net:torch.nn.Module,
                 device='cuda'):

    states = []
    actions = []
    rewards = []
    none_terminal_idxs =[]
    last_states= []
    for idx , exp in enumerate(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)

        if exp.last_state is not None:
            none_terminal_idxs.append(idx)
            last_states.append(np.array(exp.last_state,copy=False))
    
    states_t = torch.FloatTensor(np.array(states,copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    rewards = np.array(rewards,dtype=np.float32)
    with torch.no_grad():
        if none_terminal_idxs:
            # 1-get V(last_states)
            # 2-update rewards
            last_states_t = torch.FloatTensor(np.array(last_states,copy=False)).to(device)

            last_states_vals = net(last_states_t)[0].cpu().numpy()[:,1]
            last_states_vals = last_states_vals*(Hyperparams.GAMMA)**Hyperparams.REWARDS_STEPS

            rewards[none_terminal_idxs]+=last_states_vals
    q_vals_t = torch.FloatTensor(rewards).to(device)

    return states_t,actions_t,q_vals_t


# Context manager to handle the end of each episode
class EpisodeTracker:
    def __init__(self,reward_thereshold:float,
                 writer):
        self.writer = writer
        self.reward_thereshold = reward_thereshold

    def __enter__(self):
        self.ep_start_time = time.time()
        self.ep_start_frame = 0
        self.rewards = []
        return self
    def __exit__(self):
        self.writer.close()

    def episode_end(self,reward,frame):
        # write speed , reward , reward mean
        self.rewards.append(reward)        
        speed = (frame-self.ep_start_frame)/(time.time()-self.ep_start_time)

        self.ep_start_frame = frame

        reward_mean = np.mean(self.rewards[-100:])

        print(f'~~~ Episode {len(self.rewards)} ~~~')
        print(f'-> Avergae reward : {reward_mean}')
        print(f'-> Episode reward : {reward}')

        sys.stdout.flush()


        self.writer.add_scalar('speed',speed,frame)
        self.writer.add_scalar('reward',reward,frame)
        self.writer.add_scalar('reward_mean',reward_mean,frame)

        self.ep_start_time = time.time()

        if reward_mean > self.reward_thereshold:
            print(f'Solved Pong in {frame} frames !')
            return True
        return False

