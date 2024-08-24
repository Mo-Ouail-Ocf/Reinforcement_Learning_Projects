from dataclasses import dataclass
import typing as tt
import numpy as np
import torch
import torch.nn as nn

import ptan.ignite as ptan_ignite
from ptan.experience import ExperienceFirstLast, \
    ExperienceSourceFirstLast, PrioritizedReplayBuffer 

from ignite.engine import Engine ,Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger , OutputHandler
from ignite.metrics import RunningAverage

from datetime import timedelta, datetime


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


class AnnealingBetaSchedule:
    def __init__(self,beta_start:float,beta_frames:float):
        self.beta_start  = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
    def update_beta(self,iteration):
        new_val = self.beta_start + iteration *(1.0 -self.beta_start )/ self.beta_frames 
        self.beta = min(1.0,new_val)
        return self.beta



def gen_batch(buffer:PrioritizedReplayBuffer,params:HyperParameters,beta_sch:AnnealingBetaSchedule):
    buffer.populate(params.replay_initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(params.batch_size,beta=beta_sch.beta)


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



def attach_ignite(exp_source:ExperienceSourceFirstLast,trainer:Engine,parmas:HyperParameters,
                  beta_scheduler:AnnealingBetaSchedule):
    
    # attach handlers
    end_ep_hanlder = ptan_ignite.EndOfEpisodeHandler(
        exp_source=exp_source,
        bound_avg_reward=parmas.stop_reward,    
    )   
    end_ep_hanlder.attach(trainer)

    fps_handler = ptan_ignite.EpisodeFPSHandler()
    fps_handler.attach(trainer)

    periodic_events_handler = ptan_ignite.PeriodicEvents()
    periodic_events_handler.attach(trainer)

    # Logging :

    @trainer.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def handle_end_of_episode(engine):
        metrics = trainer.state.metrics # reward , steps ,avg_reward , avg_steps
        passed = metrics.get('time_passed', 0)
        print(f'~~~~ Episode {engine.state.episode} ~~~~~~~')
        print(f'->Number of steps : {metrics['steps']}')
        print(f'->Reward : {metrics['reward']}')
        print(f'->Average reward : {metrics['avg_reward']}')
        print(f'->Passed : {timedelta(seconds=int(passed))}')

    @trainer.on(ptan_ignite.EpisodeEvents.BEST_REWARD_REACHED)
    def handle_bound_reached(engine):
        metrics = trainer.state.metrics
        print('~~~~ Best reward reached ~~~~~~~')
        print(f'-> Average ReEward : {metrics['avg_reward']}')


    @trainer.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def end_train(engine):
        metrics = trainer.state.metrics
        passed = metrics.get('time_passed', 0)
        print('~~~~ Solved the game ! ~~~~~~~')
        print(f'->Number of steps : {metrics['steps']}')
        print(f'-> Time passed : {passed}')
        print(f'-> Total iterations : {trainer.state.iteration}')
        trainer.should_terminate=True
        trainer.state.solved = True

    # scheduling

    @trainer.on(Events.ITERATION_COMPLETED)
    def schedule_beta(engine):
        beta_scheduler.update_beta(trainer.state.iteration)

    # metric,
    loss_avg_metric = RunningAverage(output_transform=lambda a:a['loss'])
    loss_avg_metric.attach(trainer,name="avg_loss")

    # TB Logging

    tb_logger = TensorboardLogger(log_dir="space-invaders")
    
    end_ep_logger = OutputHandler(
        metric_names=['reward','steps','avg_reward'],
        tag="episodes",
    )
    event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb_logger.attach(
        trainer,
        end_ep_logger,
        event_name,
    )

    metrics = ['avg_loss', 'avg_fps','snr_1','snr_2']
    engine_output_logger = OutputHandler(
        output_transform= lambda a : a ,
        metric_names=metrics,
        tag="train"
    )
    
    event_name = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb_logger.attach(
        trainer,
        engine_output_logger,
        event_name,
    )





