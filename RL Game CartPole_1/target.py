import torch

def update_target_network(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())
