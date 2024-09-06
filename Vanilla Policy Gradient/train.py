import torch.nn as nn
import torch
from tensorboardX import SummaryWriter

import gymnasium as gym
import numpy as np
from ptan.agent import PolicyAgent
from ptan.experience import ExperienceSourceFirstLast
import ptan

import torch.nn.functional as F

import typing as tt

class Hyperparams:
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    ENTROPY_BETA = 0.01
    BATCH_SIZE = 8
    # how many steps ahead the Bellman equation is unrolled 
    # to estimate the discounted total reward
    REWARD_STEPS = 10 

    REWARDS_THERESHOLDS = 450


# The model
class VanillaPolicyGradient(nn.Module):
    def __init__(self,input_size,nb_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,nb_actions)
        )

    def forward(self,x):
        return self.net(x)
    
def smooth(old: tt.Optional[float], val: float,
           alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val
    

def train_pg_agent():

    env = gym.make('CartPole-v1')
    writer = SummaryWriter(logdir='vpg_cartpole')

    pg_net = VanillaPolicyGradient(env.observation_space.shape[0],
                                   env.action_space.n)
    
    optimizer = torch.optim.Adam(pg_net.parameters(), lr=Hyperparams.LEARNING_RATE)

    agent = PolicyAgent(pg_net,apply_softmax=True,
                        preprocessor=ptan.agent.float32_preprocessor)

    exp_source = ExperienceSourceFirstLast(env,agent,gamma=Hyperparams.GAMMA,
                                           steps_count=Hyperparams.REWARD_STEPS)
    
    actions = []
    states = []
    batch_scales = [] # (q_val - baseline)

    nb_episodes = 0

    total_ep_reward = 0.0 # we need it to calculate the baseline : moving avergae

    total_rewards=[] # over all episodes

    batch_scales_smoothed =  smoothed_entropy = smoothed_total_loss = smoothed_average_reward_loss =0.0


    for idx , exp in enumerate(exp_source):

        total_ep_reward+=exp.reward
        baseline = total_ep_reward/(idx+1)

        actions.append(exp.action)
        states.append(exp.state)
        batch_scales.append(exp.reward-baseline)


        rewards  = exp_source.pop_total_rewards()

        if rewards:
            nb_episodes+=1
            #check convergence condition
            reward = rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))

            print(f'- Epsiode {nb_episodes} , Reward : {reward} , Reward mean : {mean_rewards} \n')

            writer.add_scalar("reward", reward, idx)
            writer.add_scalar("reward_100", mean_rewards, idx)
            writer.add_scalar("episodes", nb_episodes, idx)

            if mean_rewards > Hyperparams.REWARDS_THERESHOLDS:
                print("Solved in %d steps and %d episodes!" % (idx, nb_episodes))
                torch.save(pg_net,f='pg_net_weights')
                break

        if len(actions) < Hyperparams.BATCH_SIZE:
            continue

        # Calculate the loss : 
        actions_t = torch.tensor(actions,dtype=torch.int)

        states_t = torch.as_tensor(np.array(states,copy=False))

        batch_scales_t = torch.tensor(batch_scales)

        logits = pg_net(states_t)

        all_log_probs = F.log_softmax(logits,dim=1)
        log_probs = all_log_probs[range(Hyperparams.BATCH_SIZE), actions_t]

        scaled_log_probs = log_probs*batch_scales_t

        expected_reward_loss = -scaled_log_probs.mean()

        # Entropy bonus
        prob_t = F.softmax(logits, dim=1)
        entropy = - (all_log_probs * prob_t).sum(dim=1).mean()

        loss = expected_reward_loss + Hyperparams.ENTROPY_BETA*entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Logging different metrics :

        with torch.no_grad():

            # 1- KL Divergence (Old || New) :
            new_logits = pg_net(states_t)
            new_probs = F.softmax(new_logits,dim=1)

            kl_divergence =- (prob_t * (prob_t/new_probs).log()).sum(dim=1).mean()

            writer.add_scalar("kl", kl_divergence.item(), idx)

            # 2- Gradients :

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in pg_net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1

            batch_scales_smoothed = smooth(
                batch_scales_smoothed,
                float(np.mean(batch_scales))
            )
        # batch_scales_smoothed =  
        # smoothed_entropy = 
        # smoothed_total_loss =
        #  smoothed_average_reward_loss =0.0

            smoothed_entropy = smooth(smoothed_entropy, entropy.item())
            smoothed_average_reward_loss = smooth(smoothed_average_reward_loss, expected_reward_loss.item())
            smoothed_total_loss = smooth(smoothed_total_loss, loss.item())

            writer.add_scalar("entropy", entropy, idx)
            writer.add_scalar("loss_entropy", smoothed_entropy, idx)
            writer.add_scalar("loss_policy", smoothed_average_reward_loss, idx)
            writer.add_scalar("loss_total", smoothed_total_loss, idx)
            writer.add_scalar("grad_l2", grad_means / grad_count,
                         idx)
            writer.add_scalar("grad_max", grad_max, idx)
            writer.add_scalar("batch_scales", batch_scales_smoothed, idx)

            states.clear()
            actions.clear()
            batch_scales.clear()
    writer.close()


if __name__=="__main__":
    train_pg_agent()