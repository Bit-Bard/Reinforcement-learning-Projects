import random
import torch

def select_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)  # explore
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()  # exploit
