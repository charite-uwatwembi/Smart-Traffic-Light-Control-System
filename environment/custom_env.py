import gym
from gym import spaces
import numpy as np

class SmartTrafficEnv(gym.Env):
    def __init__(self):
        super(SmartTrafficEnv, self).__init__()
        
        # Define action space (4 choices: open NS, open EW, keep current, prioritize pedestrians)
        self.action_space = spaces.Discrete(4)
        
        # Define state space (cars waiting in 4 lanes, traffic speed, time of day, pedestrian request)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([10, 10, 10, 10, 60, 2, 1]),
                                            dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        # Reset environment state
        self.state = np.array([np.random.randint(0, 10) for _ in range(4)] +  
                              [np.random.uniform(0, 60), np.random.randint(0, 3), np.random.randint(0, 2)])
        return self.state
    
    def step(self, action):
        reward = 0
        done = False
        
        # Get number of cars in each lane
        north, south, east, west, speed, time_of_day, pedestrian_request = self.state
        
        # Define logic for each action
        if action == 0:  # Keep current light phase
            reward -= 1  # Slight penalty for not optimizing
        elif action == 1:  # Open NS
            if north > 0 and south > 0:
                reward += 10  # Reward for effective traffic flow
            else:
                reward -= 5  # Penalty for inefficient decision
        elif action == 2:  # Open EW
            if east > 0 and west > 0:
                reward += 10
            else:
                reward -= 5
        elif action == 3:  # Give priority to pedestrians
            if pedestrian_request == 1:
                reward += 5
            else:
                reward -= 3
        
        # Simulate some vehicles passing through (reduce count)
        self.state[:4] = np.maximum(self.state[:4] - np.random.randint(0, 3, size=4), 0)
        
        # Collision penalty
        if (action == 1 and action == 2):  # Both directions open simultaneously
            reward -= 10  # Collision penalty
        
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"State: {self.state}, Action: {self.action_space}")
