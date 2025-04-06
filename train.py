import gym
from environment.custom_env import SmartTrafficEnv
from training.dqn_training import train_dqn
from training.ppo_training import train_ppo

# Create environment
env = SmartTrafficEnv()

# Train DQN
print("Training DQN...")
dqn_model = train_dqn(env, episodes=1000)

# Train PPO
print("\nTraining PPO...")
ppo_model = train_ppo(env, episodes=1000)

# Zip models (optional)
import zipfile
import os

def zip_models():
    with zipfile.ZipFile('models/dqn_ppo_models.zip', 'w') as zipf:
        for root, _, files in os.walk('models'):
            for file in files:
                zipf.write(os.path.join(root, file))
                
zip_models()
print("Models zipped successfully!")