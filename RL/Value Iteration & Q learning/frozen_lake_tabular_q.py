import gymnasium as gym
import typing as tt
from collections import defaultdict ,Counter
from torch.utils.tensorboard.writer import SummaryWriter


ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
REWARD_THRESHOLD=0.8
LEARNING_RATE = 0.2
# Types :
State = int
Action = int
ValuesMatrixKey = tt.Tuple[State,Action]

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state ,_ = self.env.reset()
        # tables
        self.values_matrix:tt.Dict[ValuesMatrixKey,float]=\
            defaultdict(float)
        
    # Updating & sampling
    def sample_step(self)->tt.Tuple[State,Action,float,State]:
        action = self.env.action_space.sample()
        new_obs , reward , is_done , is_trunc , _ = self.env.step(action)
        old_state = self.state
        out = (old_state,action,float(reward),new_obs)
        if is_done or is_trunc:
            self.state ,_=self.env.reset()
        else:
            self.state=new_obs
        return out
    

    # Testing
    def get_best_value_best_action(self,state:State)->tt.Tuple[float,Action]:
        best_action,best_value = None,None
        for action in range(self.env.action_space.n):
            action_value = self.values_matrix[(state,action)]
            if best_value is None or action_value>best_value:
                best_action=action
                best_value=action_value
        return best_value,best_action

    def value_update(self,state:State,action:Action,new_state:State,reward:float):
        best_value , _ = self.get_best_value_best_action(new_state)
        new_value = reward+GAMMA*best_value
        old_value =self.values_matrix[(state,action)]
        self.values_matrix[(state,action)] = (1-LEARNING_RATE)*old_value+LEARNING_RATE*new_value

    # for test & evaluation , we dont update values matrix for sure 
    def play_episode(self,env:gym.Env)->float:
        # returnes the reward of the episode
        obs ,_=env.reset() 
        total_reward =0.0
        while True:
            _,action = self.get_best_value_best_action(obs)
            new_obs , local_reward , is_done , is_trunc,_=env.step(action)
            # update the reward
            total_reward+=local_reward
            obs=new_obs
            if is_done or is_trunc:
                break
        return total_reward

                
if __name__=="__main__":
    # Algo
    '''
    For each step:
        1- Interact once with environement -> get (s,a,s',r)
        2- update with the blending technique the value Q(s,a)
        3- Test the updated values matrix (we pick a for which Q(s,a) is the biggest)
        4- If we reached a threshold : stop
    '''
    env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(log_dir="logs")
    iter_no=1
    best_reward = 0.0
    while True:
        state ,action , reward , new_state =agent.sample_step()
        agent.value_update(state,action,new_state,reward)
        #test
        reward=0.0
        for _ in range(TEST_EPISODES):
            reward+=agent.play_episode(env)
        reward/=TEST_EPISODES
        writer.add_scalar("reward",reward,iter_no,)
        if (reward>best_reward):
            best_reward=reward
            print((
                f"~~~~~~   Iteration {iter_no}   ~~~~~~~~~~"
                f"Reward mean : {reward}"
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            ))

        if(reward>REWARD_THRESHOLD):
            print('SOLVED !')
            break
        iter_no+=1
    writer.close()