import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import typing as tt
from dataclasses import dataclass
from lib import wrappers 
from lib import model
import time
from torch.utils.tensorboard.writer import SummaryWriter
ENV_NAME="PongNoFrameskip-v4"
# Hyperparams
GAMMA = 0.99
MEAN_BOUNDARY =19
'''epsilon greey'''
EPSILON_DECAY=150000
EPSILON_START=1.0
EPSILON_FINAL=0.1

LEARNING_RATE = 1e-4
BATCH_SIZE =32

'''REPLAY BUFF'''
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000

SYNC_TARGET_NN = 1000 # How frequently we sync DQN & TNN

State = np.ndarray
Action = int

BatchTensors = tt.Tuple[
    torch.ByteTensor,#initial state
    torch.LongTensor,# actions
    torch.Tensor, # rewards
    torch.BoolTensor,# is_done or is_trunc
    torch.ByteTensor # final state
]

@dataclass
class Experience: #(s,a,r,b,s')
    state:State
    action:Action
    reward:float
    is_done_trunc:bool
    new_state:State

class ReplayBuffer:
    def __init__(self,buff_size:int):
        self.buffer = deque(maxlen=buff_size)

    def append(self,exp:Experience):
        self.buffer.append(exp)


    def __len__(self):
        return len(self.buffer)
    
    def sample(self,batch_sizs:int)->tt.List[Experience]:
        random_idxs = np.random.choice(len(self),batch_sizs,replace=False)
        batch = [self.buffer[idx] for idx in random_idxs]
        return batch
    
class Agent:
    def __init__(self,env:gym.Env,exp_buffer:ReplayBuffer):
        self.env = env
        self.experience_buffer = exp_buffer
        self.state : tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state , _ =self.env.reset()
        self.total_reward =0.0

    # interact with env to add new experience , returns total reward if ep ends
    @torch.no_grad
    def play_step(self,model:model.DQNAgent,
                  epsilon:float,device:torch.device)->tt.Optional[float]:
        
        total_reward = None
        # pick an action !
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # get action by prediction
            state_v = torch.as_tensor(self.state).to(device).unsqueeze(0)
            pred = model(state_v)
            _ , action_indice = torch.max(pred,dim=1)
            action = int(action_indice.item())
        # Execute the action
        new_state , reward , is_done , is_trunc ,_ =self.env.step(action)
        exp = Experience(
            state=self.state,
            action=action,
            reward=reward,
            is_done_trunc=is_done or is_trunc,
            new_state=new_state
        )        
        self.experience_buffer.append(exp)
        self.total_reward+=reward
        self.state=new_state
        if exp.is_done_trunc:
            total_reward=self.total_reward
            self._reset()
        return total_reward
    


def convert_batch_to_tensors(batch:tt.List[Experience],device:torch.device)->BatchTensors:
    #batch : [exp1,...,exp_k]
    # exp [s,a,r,s',b]
    states , actions , rewards , stops ,new_states = [],  [],  [],  [],  []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        stops.append(e.is_done_trunc)
        new_states.append(e.new_state)
    states_v = torch.as_tensor(np.array(states,copy=False)).to(device) #avoid copying the data if possible & keep same type
    new_states_v = torch.as_tensor(np.array(new_states,copy=False)).to(device) 
    stops_v = torch.BoolTensor(stops).to(device)
    rewards_v=torch.FloatTensor(rewards).to(device)
    actions_v = torch.LongTensor(actions).to(device)


    return states_v , actions_v , rewards_v , stops_v ,new_states_v


def calc_loss(model:model.DQNAgent,
              target_net:model.DQNAgent,
              batch:tt.List[Experience],
              device:torch.device)->torch.Tensor:
    states_v , actions_v , rewards_v , stops_v ,new_states_v = \
        convert_batch_to_tensors(batch,device)
    # Calculate targets
    pred_q_values = model(states_v).gather(
        1,actions_v.unsqueeze(-1)
    ).squeeze(-1) # 1d array of q values

    with torch.no_grad(): # to not affect 2nd network
        next_state_action_values = \
            target_net(new_states_v).max(1)[0] # get Max Q(s',a) for a in A
        next_state_action_values[stops_v]=0.0 # for finished states ; Q = 0.0
        next_state_action_values=next_state_action_values.detach()

    updated_q_values = GAMMA*next_state_action_values+rewards_v
    return nn.MSELoss()(updated_q_values,pred_q_values)


if __name__=="__main__":
    writer = SummaryWriter()
    device = torch.device("cuda")

    env = wrappers.make_env(ENV_NAME)
    input_shape = env.observation_space.shape
    nb_actions = int(env.action_space.n)
    buffer = ReplayBuffer(REPLAY_SIZE)

    # the model
    net = model.DQNAgent(input_shape,nb_actions).to(device)
    target_net=model.DQNAgent(input_shape,nb_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

    agent = Agent(env=env,exp_buffer=buffer)

    frame_idx =0
    ep_start_idx=0

    rewards_per_episode =[]

    best_m_reward = None

    epsilon = EPSILON_START
    
    time_start_ep = time.time()

    while True:
        frame_idx+=1
        epsilon =max(EPSILON_FINAL,EPSILON_START-frame_idx/EPSILON_DECAY)
        reward = agent.play_step(net,epsilon=epsilon,device=device)

        if reward is not None:
            rewards_per_episode.append(reward)
            mean_reward = np.mean(rewards_per_episode[-100:])
            # end of episode : 
            speed = (frame_idx-ep_start_idx) /( time.time()-time_start_ep)
            ep_start_idx = frame_idx
            time_start_ep=time.time()
            print((
                "******************************\n"
                f"---> Frame :{frame_idx} \n"
                f"---> Reward for episode {len(rewards_per_episode)} :{reward} \n"
                f"---> Reward for last 100 episode : {mean_reward} \n"
            ))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < mean_reward:
                torch.save(net.state_dict(),f='model.ph')
                if best_m_reward is not None:
                    print((             
                        f"---> Best reward updated "
                        f"{best_m_reward:.3f} -> {mean_reward:.3f}"
                        "******************************\n"
                    ))
                best_m_reward =mean_reward
            if mean_reward > MEAN_BOUNDARY:
                print("Solved in %d frames!" % frame_idx)
                break
            if len(buffer)<REPLAY_START_SIZE :
                continue
            if frame_idx % SYNC_TARGET_NN == 0:
                target_net.load_state_dict(net.state_dict())
         
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss = calc_loss(net,target_net,batch,device)
            loss.backward()
            optimizer.step()
        writer.close()

           