from stable_baselines3.common import atari_wrappers

import gymnasium as gym
from gymnasium import spaces

import numpy as np
from collections import deque #for stacked frames
import typing as tt


'''
- This class returns the stacked sequential frames in ndarray
of shape [n_stacked_frames , W , H]
- We use internal stack to store the sequetial frames
- when calling reset , we return array of inital observation with low : (low,...,obs)
- when calling step(action) : new observation will be added to the stack , old one popped
'''
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env,n_stacked_frames:int):
        super().__init__(env)
        # stack array containing frames to be stacked into one state (n_stacked_frames,W,H)
        self.buffer = deque(maxlen=n_stacked_frames)
        old_space = env.observation_space
        assert isinstance(old_space,spaces.Box)
        assert len(old_space.low.shape)==3 # (1,W,H)

        new_space = spaces.Box(
            low=old_space.low.repeat(n_stacked_frames,axis=0), # (n_stacked_frames,W,H)
            high=old_space.high.repeat(n_stacked_frames,axis=0),
            dtype=old_space.dtype
        )
        self.observation_space=new_space
        # observations : (n_stacked_frames,W,H)
    def reset(self, *, seed: tt.Optional[int] = None,
              options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1):
            self.buffer.append(self.env.observation_space.low)
        new_obs ,info= self.env.reset()
            #returns the initial observation from 
            # the underlying environment, which is not yet 
            # transformed.
        return self.observation(new_obs),info
    
    def observation(self, observation:np.ndarray)->np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer) 
        '''
        
        buffer = [ (1,W,H) , ... ,(1,W,H) ]
        returns (n_stacked_frames,W,H)

        '''

    
# converts img obs into CWH
class ImgToPytorch(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = self.env.observation_space
        assert isinstance(old_space,spaces.Box)
        new_space = spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=(old_space.shape[-1],old_space.shape[0],old_space.shape[1]), 
            dtype=old_space.dtype
        )
        # obs : [Cin,W,H]
        self.observation_space=new_space

    def observation(self, observation: np.ndarray):
        new_obs = np.moveaxis(observation,2,0)
        return new_obs
    # obs & low , high match


def make_env(env_name:str):
    env = gym.make(env_name)
    env = atari_wrappers.AtariWrapper(
        env=env,
        clip_reward=False, # bcz pong retuns -1 , 0 , 1 
        noop_max=0,
    )
    env = ImgToPytorch(env)
    env = BufferWrapper(env,n_stacked_frames=4)
    return env


"""
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default
    
"""