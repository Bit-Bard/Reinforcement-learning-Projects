import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np

from dqn_model import DQN
from replay_buffer import ReplayBuffer
from policy import select_action
from train_step import train_step
from target import update_target_network

# ------------------- Hyperparameters -------------------
EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 3.3e-3
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 20

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998
# ------------------------------------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
update_target_network(policy_net, target_net)

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START

# ------------------- Training Loop -------------------
for episode in range(EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    total_reward = 0

    while True:
        action = select_action(state, policy_net, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)

        cart_pos = next_state[0]
        reward = reward - 0.01 * abs(cart_pos)

        done = terminated or truncated

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        train_step(
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            BATCH_SIZE,
            GAMMA
        )

        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
        update_target_network(policy_net, target_net)

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()

# after training loop ends
torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
print("Model saved as cartpole_dqn.pth")
