import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter

from dataclasses import dataclass

# Consts
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
device = torch.device('cpu')
@dataclass
class EpisodeStep: # (s,a)
    action:int
    observation:np.ndarray

@dataclass
class Episode: # [(s,a)] + total_reward
    total_reward:float 
    espisode_steps:tt.List[EpisodeStep]

def get_batch(model:nn.Module,
              batch_size:int,
              env:gym.Env)->tt.Generator[tt.List[Episode],None,None]:
    # convert logits to proba
    softmax = nn.Softmax(dim=1)
    # curr step
    batch = []
    # curr episode
    episode_reward = 0.0
    episode_steps =[]
    observation,_= env.reset()
    while True:
        # get logits for current action
        observation_t = torch.tensor(observation,dtype=torch.float32)
        action_logits = model(observation_t.unsqueeze(0))
        action_probas = softmax(action_logits).data.numpy()[0]

        # sample action from the distribution returned by the model
        action = np.random.choice(len(action_probas),p=action_probas)
        episode_step = EpisodeStep(action,observation)
        episode_steps.append(episode_step)

        # take new action
        observation,reward,is_done,is_trunc,_ = env.step(action)
        episode_reward+=reward
        if (is_done or is_trunc):
            episode = Episode(episode_reward,episode_steps)
            batch.append(episode)
            observation,_ = env.reset()
             # curr episode
            episode_reward = 0.0
            episode_steps =[]
            if len(batch)==batch_size:
                yield batch
                batch=[]

'''
-training of our NN and the generation of our episodes are performed at the 
same time.
-They are not completely in parallel, but every time our loop accumulates 
enough episodes (16), it passes control to trainer
-So, when yield is returned,
the NN will have different, slightly better (we hope) behavior.
'''

def filter_batch(batch:tt.List[Episode],percentile:float)-> \
        tt.Tuple[torch.FloatTensor,torch.LongTensor,float,float]:
    #get rewards array
    rewards = list(map(lambda episode : episode.total_reward,batch))
    threshold = float(np.percentile(rewards,percentile))
    mean = float(np.mean(rewards))

    train_obs_list : tt.List[np.ndarray]=[]
    train_actions:tt.List[int]=[]

    for episode in batch:
        if episode.total_reward>=threshold:
            #insert observations & corresponding actions
            train_obs_list.extend(map(lambda step:step.observation,episode.espisode_steps))
            train_actions.extend(map(lambda step:step.action,episode.espisode_steps))
    train_obs_v = torch.FloatTensor(np.vstack(train_obs_list))
    train_actions_v = torch.LongTensor(train_actions)

    return train_obs_v ,train_actions_v ,threshold ,mean

class NeuralNet(nn.Module):
    def __init__(self,obs_size:int,hidden_size:int,
                 n_actions:int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions)
        )

    def forward(self,obs:torch.Tensor):
        # returns logits array of actions
        return self.net(obs)


if __name__=="__main__": 
    env = gym.make('CartPole-v1')

    #get nn inputs
    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    model = NeuralNet(obs_size,HIDDEN_SIZE,n_actions).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(),lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter , batch in enumerate(get_batch(model,BATCH_SIZE,env)):
        obs_v ,actions_v,reward_bound ,rewards_mean = filter_batch(batch=batch,percentile=PERCENTILE)
        optim.zero_grad()
        logits = model(obs_v)
        loss = loss_fn(logits,actions_v)
        loss.backward()
        optim.step()

        print((
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            f"--> Iteration {iter} : \n"
            f"---->Loss : {loss.item()}"
            f"---->Reward mean : {rewards_mean}"
            f"---->Reward boundary : {reward_bound}"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        ))
        writer.add_scalar("loss", loss.data.item(), iter)
        writer.add_scalar("reward_bound", reward_bound, iter)
        writer.add_scalar("reward_mean", rewards_mean, iter)
        if rewards_mean > 475:
            print("Solved!")
            break
    writer.close()
    
        