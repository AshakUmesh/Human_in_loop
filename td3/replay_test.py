from td3.replay_buffer import ReplayBuffer
import numpy as np

state_dim = 27648
action_dim = 3

buffer = ReplayBuffer(state_dim, action_dim)

state = np.random.randn(state_dim)
action = np.random.randn(action_dim)

buffer.add(state, action, 1, state, 0)

batch = buffer.sample(1)

print("Sampled state shape:", batch[0].shape)