

class Hyperparams:

    LEARNING_RATE = 0.003
    GAMMA = 0.99

    BATCH_SIZE = 128
    ENTROPY_BETA = 0.01
    NUM_ENVS = 50

    REWARDS_STEPS = 4 # unrolling Bellman equation to estimate Q Value

    CLIP_GRAD = 0.1 # if L2 nrom > CLIP_GRAD : clip grad vec to this value

    REWARD_THRESHOLD = 20 

ENV_NAME = 'PongNoFrameskip-v4'