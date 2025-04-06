import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim))
        
    def forward(self, x):
        return self.net(x)

def train_dqn(env, episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    online_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)
    epsilon = epsilon_start
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = online_net(state_tensor).argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch = list(zip(*transitions))
                
                states = torch.FloatTensor(np.array(batch[0])).to(device)
                actions = torch.LongTensor(batch[1]).to(device)
                rewards = torch.FloatTensor(batch[2]).to(device)
                next_states = torch.FloatTensor(np.array(batch[3])).to(device)
                dones = torch.FloatTensor(batch[4]).to(device)
                
                current_q = online_net(states).gather(1, actions.unsqueeze(1))
                next_q = target_net(next_states).max(1)[0].detach()
                target = rewards + gamma * next_q * (1 - dones)
                
                loss = nn.MSELoss()(current_q.squeeze(), target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_net.load_state_dict(online_net.state_dict())
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")
        
        if episode % 100 == 0:
            os.makedirs("models/dqn", exist_ok=True)
            torch.save(online_net.state_dict(), f"models/dqn/dqn_{episode}.pth")
    
    return online_net