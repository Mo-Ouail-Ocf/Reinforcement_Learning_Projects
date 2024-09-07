import torch.nn as nn
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from lib.model import ActorCriticNet
from lib.config import Hyperparams ,ENV_NAME
import ptan
from lib.utils import unpack_batch,EpisodeTracker
from ptan.common.wrappers import wrap_dqn
import gymnasium as gym
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.agent import PolicyAgent
from ptan.common.utils import TBMeanTracker
import numpy as np
from tensorboardX import SummaryWriter

import torch
import os

SAVE_EVERY = 10_000

def save_model_and_optimizer(model, optimizer, frame, save_dir='weights', save_every=SAVE_EVERY):
    """
    Save the model and optimizer state_dict every 'save_every' frames.
    """
    if frame % save_every == 0:
        filename = os.path.join(save_dir, f'a2c_weights_{frame}.pth')

        # Remove old checkpoints if they exist
        previous_frame = frame - save_every
        if previous_frame > 0:
            old_filename = os.path.join(save_dir, f'a2c_weights_{previous_frame}.pth')
            if os.path.exists(old_filename):
                os.remove(old_filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'frame': frame,
        }, filename)

def test_write():
    import time

    # Example model and optimizer
    net = ActorCriticNet((4,64,64),4).to('cuda')
    optimizer = torch.optim.Adam(net.parameters(),lr=Hyperparams.LEARNING_RATE,eps=1e-3)

    # Measure time for saving
    start_time = time.time()

    save_model_and_optimizer(net, optimizer, frame=510000)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken to save model: {execution_time:.4f} seconds")


def train():


    env_factories = [
            lambda: ptan.common.wrappers.wrap_dqn(
                gym.make("PongNoFrameskip-v4"))
            for _ in range(Hyperparams.NUM_ENVS)
    ]

    env = gym.vector.SyncVectorEnv(env_factories)

    net = ActorCriticNet(env.single_observation_space.shape,
                          env.single_action_space.n).to('cuda')
    optimizer = torch.optim.Adam(net.parameters(),lr=Hyperparams.LEARNING_RATE,eps=1e-3)

    agent = PolicyAgent(lambda x:net(x)[0] , device='cuda',apply_softmax=True)

    writer = SummaryWriter(logdir='pong_a2c_agent')

    exp_source = VectorExperienceSourceFirstLast(env=env,agent=agent,gamma=Hyperparams.GAMMA,steps_count=Hyperparams.REWARDS_STEPS)

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
                policy_loss = -((log_probas_t*advantage_t).mean())


                ####### Entropy loss = E [ log(pi(a|s)) ]
                entropy_loss = Hyperparams.ENTROPY_BETA*(all_probas_t*all_log_probas_t).sum(dim=1).mean()

                ###### Value loss = E[ (q_vals_t - values_t )**2 ] 
                value_loss = F.mse_loss(values_t.squeeze(-1),q_vals_t)

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

                save_model_and_optimizer(net,optimizer,idx)


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
                    'q_vals':q_vals_t.detach().cpu(),
                    'advantage':advantage_t.detach().cpu(),
                    'values':values_t.detach().cpu(),
                }

                for param_name , param_value in to_log.items():
                    mean_tracker.track(
                    param_name, param_value, idx)


               

if __name__=="__main__":
    train()