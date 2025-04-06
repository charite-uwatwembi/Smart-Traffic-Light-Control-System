import gymnasium as gym
import torch
import numpy as np
import time
from environment.custom_env import SmartTrafficEnv
from training.dqn_training import DQN

MODEL_PATH = "models/dqn/dqn_900.pth"

# Initialize environment
env = SmartTrafficEnv()

# Load trained DQN model
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
output_dim = env.action_space.n
dqn = DQN(input_dim, output_dim)
dqn.load_state_dict(torch.load(MODEL_PATH))
dqn.eval()

# Run simulation
state, _ = env.reset()
state = torch.tensor(state.flatten(), dtype=torch.float32)

done = False
while not done:
    action = torch.argmax(dqn(state)).item()
    next_state, reward, done, _, _ = env.step(action)
    
    env.render()  # Use the environment's render method

    state = torch.tensor(next_state.flatten(), dtype=torch.float32)
    time.sleep(0.1)

env.close()  # Properly close the environment's renderer