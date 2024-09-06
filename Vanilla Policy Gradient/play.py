import ptan
import torch
import gymnasium as gym
from ptan.agent import PolicyAgent
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
import numpy as np

from train import VanillaPolicyGradient  # Assuming this is where your model is defined
def play(checkpoint_path: str):
    # Set up the environment
    env = gym.make('CartPole-v1', render_mode='human')
    env.metadata['render_fps'] = 30

    # Load the model architecture
    model = VanillaPolicyGradient(env.observation_space.shape[0], env.action_space.n)
    
    # Load the saved state_dict
    model=(torch.load(checkpoint_path))
    model.eval()  # Put the model in evaluation mode

    agent = PolicyAgent(model, apply_softmax=True,preprocessor=ptan.agent.float32_preprocessor)
    env.reset()

    total_reward = 0.0
    done = False
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=0.1)
    buff = ExperienceReplayBuffer(exp_source, 100)

    while True:
        buff.populate(1)
        env.render()

        for reward, steps in exp_source.pop_rewards_steps():
            total_reward += reward
            print(f"Total reward: {total_reward}")

        if done:
            break

if __name__ == "__main__":
    play('pg_net_weights')
