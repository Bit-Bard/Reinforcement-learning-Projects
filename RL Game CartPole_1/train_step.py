import torch
import torch.nn.functional as F

def train_step(
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    batch_size,
    gamma
):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    q_values = policy_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = F.mse_loss(q_value, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
