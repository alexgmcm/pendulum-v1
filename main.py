import gymnasium as gym

env = gym.make("Pendulum-v1", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("obs:", obs, "reward:", reward)

input("Press Enter to close...")
env.close()