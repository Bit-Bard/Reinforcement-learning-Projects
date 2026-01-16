import gymnasium as gym
import torch
from dqn_model import DQN
env = gym.make("CartPole-v1", max_episode_steps=1000, render_mode="human")


model = DQN(4, 2)
model.load_state_dict(torch.load("cartpole_dqn.pth"))
model.eval()

state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

while True:
    with torch.no_grad():
        action = torch.argmax(model(state)).item()

    next_state, _, terminated, truncated, _ = env.step(action)
    state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    if terminated:
        print("Pole fell / constraint violated")
        print("State:", state.numpy())
        break

    if truncated:
        print("Time limit reached (success)")
        break



env.close()