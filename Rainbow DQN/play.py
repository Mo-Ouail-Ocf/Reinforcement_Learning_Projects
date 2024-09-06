import torch
import gymnasium as gym
from ptan.agent import DQNAgent, TargetNet
from ptan.common.wrappers import wrap_dqn
from lib.model import RainbowDQN
from lib.utils import HyperParameters,GAME_PARAMS
from ptan.actions import ArgmaxActionSelector
from ptan.experience import ExperienceSourceFirstLast ,ExperienceReplayBuffer

def play(params: HyperParameters, checkpoint_path: str, device='cuda'):
    # Set up the environment
    env = gym.make(params.env_name,render_mode='human')
    env.metadata['render_fps'] = 30 
    env = wrap_dqn(env)

    # Load the model
    model = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Set up the agent
    action_selector = ArgmaxActionSelector()
    agent = DQNAgent(model,action_selector, device=device)
    env.reset()
    # Run the game
    total_reward = 0.0
    done = False
    exp_source = ExperienceSourceFirstLast(env,agent,gamma=0.1)
    buff = ExperienceReplayBuffer(exp_source,100)
    while True:
        buff.populate(1)
        env.render()
        for rewrad ,steps in exp_source.pop_rewards_steps():
            print(f"Total reward: {rewrad}")

if __name__ == "__main__":
    # Assuming you are loading the checkpoint from 'models/model_10000.pth'
    play(GAME_PARAMS['invaders'], checkpoint_path='./models_invaders/model_checkpoint_620000.pt')
