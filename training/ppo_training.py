import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        
    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        for _ in range(10):  # Number of optimization epochs
            new_logits, values = self.model(states)
            dist = Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(actions)
            
            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            entropy_loss = -dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
def train_ppo(env, episodes=1000, max_steps=200, update_interval=2048):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    ppo = PPO(input_dim, output_dim)
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        for _ in range(update_interval):
            state_tensor = torch.FloatTensor(state).to(ppo.device)
            with torch.no_grad():
                logits, value = ppo.model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            
            state = next_state
            if done:
                state = env.reset()
        
        returns = []
        advantages = []
        gae = 0
        last_value = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[i+1]
            delta = rewards[i] + ppo.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + ppo.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        ppo.update(states, actions, log_probs, returns, advantages)
        
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
        
        if episode % 100 == 0:
            os.makedirs("models/ppo", exist_ok=True)
            torch.save(ppo.model.state_dict(), f"models/ppo/ppo_{episode}.pth")
    
    return ppo.model