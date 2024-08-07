import gymnasium as gym
import typing as tt
from collections import defaultdict ,Counter
from torch.utils.tensorboard.writer import SummaryWriter


ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
REWARD_THRESHOLD=0.8
# Types :
State = int
Action = int
RewardMatrixKey = tt.Tuple[State,Action,State]
TransitionMatrixKey = tt.Tuple[State,Action]

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state ,_ = self.env.reset()
        # tables
        self.transition_matrix : tt.Dict[TransitionMatrixKey,Counter] = \
            defaultdict(Counter)
        self.rewards_matrix:tt.Dict[RewardMatrixKey,float]=\
            defaultdict(float)
        self.values_matrix:tt.Dict[State,float]=\
            defaultdict(float)
        
    # Updating & sampling
    def play_n_random_steps(self,n:int):
        # update rewards_matrix & transition_matrix by sampling n times\
        for _ in range(n):
            action = self.env.action_space.sample()
            new_obs,reward,is_done,is_truncated,_ = self.env.step(action)
            rm_key=(self.state,action,new_obs)
            self.rewards_matrix[rm_key] = float(reward)
            tm_key=(self.state,action)
            self.transition_matrix[tm_key][new_obs]+=1
            self.state=new_obs
            if is_done or is_truncated:
                self.state,_ = self.env.reset()

    def calc_q_value(self,state:State,action:Action)->float:
        q_value =0.0
        nb_appearance = sum(self.transition_matrix[(state,action)].values())
        for new_state , count in self.transition_matrix[(state,action)].items():
            transit_proba = count/nb_appearance
            reward = self.rewards_matrix[state,action,new_state]
            state_value = self.values_matrix[new_state]
            q_value+=transit_proba*(reward+GAMMA*state_value)
        return q_value


    # Testing
    def get_best_action(self,state:State)->Action:
        best_action,best_value = None,None
        for action in range(self.env.action_space.n):
            action_value = self.calc_q_value(state,action)
            if best_value is None or action_value>best_value:
                best_action=action
                best_value=action_value
        return best_action

    def play_episode(self,env:gym.Env)->float:
        # returnes the reward of the episode
        obs ,_=env.reset() 
        total_reward =0.0
        while True:
            action = self.get_best_action(obs)
            new_obs , local_reward , is_done , is_trunc,_=env.step(action)
            # side quest : update the 2 matrices
            self.rewards_matrix[(obs,action,new_obs)]=local_reward
            self.transition_matrix[(obs,action)][new_obs]+=1
            # update the reward
            total_reward+=local_reward
            obs=new_obs
            if is_done or is_trunc:
                break
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            # update the value : get the biggest action value
            state_values = [
                self.calc_q_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values_matrix[state]=max(state_values)
        

if __name__=="__main__":
    env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(log_dir="logs")
    iter_no=1

    while True:
        agent.play_n_random_steps(100)
        # I missed calling this method ,then I spent 1 hour debugging
        agent.value_iteration()
        #test
        reward=0.0
        for _ in range(TEST_EPISODES):
            reward+=agent.play_episode(env)
        reward/=TEST_EPISODES
        writer.add_scalar("reward",reward,iter_no,)
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