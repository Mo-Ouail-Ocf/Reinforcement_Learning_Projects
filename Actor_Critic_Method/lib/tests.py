import gymnasium as gym
from utils import unpack_batch
from ptan.experience import ExperienceReplayBuffer,ExperienceSourceFirstLast
from ptan.agent import PolicyAgent
from ptan.common.wrappers import wrap_dqn
from model import ActorCriticNet

class Hyperparams:

    LEARNING_RATE = 0.001
    GAMMA = 0.99

    BATCH_SIZE = 128

    NUM_ENVS = 50

    REWARDS_STEPS = 4 # unrolling Bellman equation to estimate Q Value

    CLIP_GRAD = 0.1 # if L2 nrom > CLIP_GRAD : clip grad vec to this value

ENV_NAME = 'PongNoFrameskip-v4'

if __name__=="__main__":
    env = gym.make(ENV_NAME)
    env = wrap_dqn(env)

    net = ActorCriticNet(env.observation_space.shape,env.action_space.n).to('cuda')

    agent = PolicyAgent(lambda x:net(x)[0] , device='cuda',apply_softmax=True)


    exp_source = ExperienceSourceFirstLast(env,agent,gamma=Hyperparams.GAMMA,steps_count=Hyperparams.REWARDS_STEPS)

    batch = []

    for idx , exp in enumerate(exp_source):
        batch.append(exp)

        if idx==Hyperparams.BATCH_SIZE-1:
            break

    states_t,actions_t,q_vals_t = unpack_batch(batch,net)     
