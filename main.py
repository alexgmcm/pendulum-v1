import gymnasium as gym
from dotmap import DotMap

env = gym.make("Pendulum-v1", render_mode="human")
#obs, info = env.reset(seed=42)

def collect_random_data(env, episodes=10):
    # get data from random policy
    dataset = []
    for i in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            record = DotMap({"s":s, "a":a, "r":r, "s_next":s_next,"done":done})
            dataset.append(record)
            s = s_next

    return dataset

# TODO: train simple neural net to predict state transitons (s,a) -> (s_next - s)

# write to file for inspection
dataset = collect_random_data(env,1)
with open('dataset.txt', 'w') as f:
    for line in dataset:
        f.write(f"{line}\n")

input("Press Enter to close...")
env.close()