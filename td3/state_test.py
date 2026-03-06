import gymnasium as gym
import numpy as np

# Create environment
env = gym.make("CarRacing-v3")

# Reset environment
state, _ = env.reset()

print("Original shape:", state.shape)

# Normalize pixel values (0-255 → 0-1)
state = state / 255.0

# Flatten the image
flat_state = state.flatten()

print("Flattened shape:", flat_state.shape)

print("First 10 values after normalization:")
print(flat_state[:10])