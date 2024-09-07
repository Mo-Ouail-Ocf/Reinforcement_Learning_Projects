import torch.nn as nn
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from lib.config import Hyperparams ,ENV_NAME
from lib.model import ActorCriticNet
from lib.utils import unpack_batch,EpisodeTracker
from ptan.common.wrappers import wrap_dqn
import gymnasium as gym
from ptan.experience import ExperienceReplayBuffer,ExperienceSourceFirstLast
from ptan.agent import PolicyAgent
from ptan.common.wrappers import wrap_dqn
from ptan.common.utils import TBMeanTracker
import numpy as np
from tensorboardX import SummaryWriter

def train():


    envs = [wrap_dqn(gym.make(ENV_NAME)) for _ in Hyperparams.NUM_ENVS]

    net = ActorCriticNet(envs[0].observation_space.shape,envs[0].action_space.n).to('cuda')
    optimizer = torch.optim.Adam(net.parameters(),lr=Hyperparams.LEARNING_RATE,eps=1e-3)

    agent = PolicyAgent(lambda x:net(x)[0] , device='cuda',apply_softmax=True)

    writer = SummaryWriter(logdir='pong_a2c_agent')

    exp_source = ExperienceSourceFirstLast(env=envs,agent=agent,gamma=Hyperparams.GAMMA,steps_count=Hyperparams.REWARDS_STEPS)

    batch = []

    with EpisodeTracker(Hyperparams.REWARD_THRESHOLD,writer) as ep_tracker:
        with TBMeanTracker(writer,batch_size=10) as mean_tracker:
            for idx , exp in enumerate(exp_source):
                batch.append(exp)

                rewards = exp_source.pop_total_rewards()

                if rewards:
                    is_solved =ep_tracker.episode_end(rewards[0],idx)
                    if is_solved:
                        break
                
                if len(batch)<Hyperparams.BATCH_SIZE:
                    continue
            
                optimizer.zero_grad()
                # compute the losses :
                states_t,actions_t,q_vals_t = unpack_batch(batch,net)
                batch.clear()

                # Policy loss : E [ log(pi) * ( Q_vals - V(s)) ]

                all_logits_t ,values_t= net(states_t)

                all_log_probas_t = F.log_softmax(all_logits_t,dim=1)
                all_probas_t = F.softmax(all_logits_t,dim=1)

                log_probas_t = all_log_probas_t[range(Hyperparams.BATCH_SIZE),actions_t]

                advantage_t = q_vals_t - values_t.detach()
                ######### Policy loss = - E [ log_probas_t*advantage_t ]
                policy_loss = -(log_probas_t*advantage_t).mean()


                ####### Entropy loss = E [ log(pi(a|s)) ]
                entropy_loss = (all_probas_t*all_log_probas_t).sum(dim=1).mean()

                ###### Value loss = E[ (q_vals_t - values_t )**2 ] 
                value_loss = F.mse_loss(q_vals_t,values_t.squeeze(-1))

                ##### Backprop policy loss : 
                policy_loss.backward(retain_graph=True)

                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters() 
                    if p.grad is not None
                ])
                
                loss_t = entropy_loss+value_loss
                loss_t.backward()

                # Clip gradients of the net
                clip_grad_norm_(parameters=net.parameters(),max_norm=Hyperparams.CLIP_GRAD)

                optimizer.step()

                # Logging time ! :
                to_log = {
                    # Losses
                    'policy_loss':policy_loss.item(),
                    'entropy_loss':entropy_loss.item(),
                    'value_loss':value_loss.item(),
                    # Gradients 
                    'gradient_max':np.max(np.abs(grads)),
                    'gradient_l2':np.sqrt(np.mean(np.square(grads))),
                    'grad_var':np.var(grads),
                    # advantage , values , q_vals 
                    'q_vals':q_vals_t,
                    'advantage':advantage_t,
                    'values':values_t
                }

                for param_name , param_value in to_log.items():
                    mean_tracker.track(
                    param_name, param_value, idx)

               

if __name__=="__main__":
    train()