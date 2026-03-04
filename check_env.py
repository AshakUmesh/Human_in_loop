import gymnasium as gym

env = gym.make("CarRacing-v3")

print("Observation shape:", env.observation_space.shape)
print("Action space:", env.action_space)