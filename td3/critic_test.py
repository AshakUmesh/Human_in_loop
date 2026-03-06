import torch
from td3.critic import Critic

state_dim = 27648
action_dim = 3

critic = Critic(state_dim, action_dim)

state = torch.randn(1, state_dim)
action = torch.randn(1, action_dim)

q1, q2 = critic(state, action)

print("Q1:", q1)
print("Q2:", q2)