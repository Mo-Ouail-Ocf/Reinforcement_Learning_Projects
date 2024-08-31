from lib.model import RainbowDQN
from lib.utils import attach_ignite,calc_loss,gen_batch,HyperParameters,GAME_PARAMS,AnnealingBetaSchedule
from ignite.engine import Engine,Events
from ignite.handlers import Checkpoint, DiskSaver

import torch
import ptan.ignite as ptan_ignite
from ptan.agent import TargetNet

from ptan.agent import DQNAgent
from ptan.experience import ExperienceSourceFirstLast , PrioritizedReplayBuffer
from ptan.common.wrappers import wrap_dqn
from ptan.actions import ArgmaxActionSelector

import gymnasium as gym

ALPHA = 0.6
N_STEPS = 2
# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 500_000

PERIODIC_LOGGING =100


def train(params:HyperParameters,device='cuda'):
    
    env = gym.make(params.env_name)
    env = wrap_dqn(env)

    action_selector = ArgmaxActionSelector()
    model = RainbowDQN(env.observation_space.shape,env.action_space.n).to(device)
    target_net = TargetNet(model)
    target_net.target_model = target_net.target_model.to('cuda') 
    agent = DQNAgent(model,action_selector,device='cuda')

    exp_source = ExperienceSourceFirstLast(env,agent,gamma=params.gamma,
                                           steps_count=N_STEPS)
    
    priority_buff = PrioritizedReplayBuffer(
        alpha=ALPHA,
        buffer_size=params.batch_size,
        experience_source=exp_source,
    )

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=params.learning_rate)


    def process_batch(engine:Engine,batch):
        transitions, idxes, weights = batch
        optimizer.zero_grad()

        model.reset_noise()
        loss , new_priors = calc_loss(target_net.target_model,model,transitions,weights,
                                      params.gamma**N_STEPS)
        
        loss.backward()

        optimizer.step()

        priority_buff.update_priorities(idxes,new_priors)

        if engine.state.iteration % params.target_net_sync ==0 :
            target_net.sync()
            target_net.target_model = target_net.target_model.to('cuda') 


        if engine.state.iteration % PERIODIC_LOGGING == 0:
            snrs = model.noisy_layers_snr()
            engine.state.metrics['snr_1'] = snrs[0]  
            engine.state.metrics['snr_2'] = snrs[1]

        return {
            'loss':loss.item(),
        }
    
    beta_schedule = AnnealingBetaSchedule(BETA_START,BETA_FRAMES)

    trainer = Engine(process_batch)


    checkpoint_handler = Checkpoint(
        to_save={'model': model, 'optimizer': optimizer},  # Save both model and optimizer
        save_handler=DiskSaver('models', create_dir=True),  # Directory to save the models
        n_saved=3,  # Number of checkpoints to keep
        filename_prefix='model',  # Prefix for the checkpoint filenames
        global_step_transform=lambda engine, event: engine.state.iteration  # Naming based on iterations
    )


    attach_ignite(exp_source,trainer,params,beta_schedule)

    trainer.add_event_handler(ptan_ignite.PeriodEvents.ITERS_10000_COMPLETED, checkpoint_handler)

    state = trainer.run(gen_batch(priority_buff,params,beta_schedule))



if __name__=="__main__":
    train(GAME_PARAMS['invaders'])

    



        


    

    