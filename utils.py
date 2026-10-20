
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np


def record_agent(
    model_path="q_table.npy",
    output_path="trained_agent.gif",
    seed=123,
):
    # Load the trained Q-table
    q_table = np.load(model_path)

    # Render environment as RGB frames
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="rgb_array",
    )

    state, info = env.reset(seed=seed)
    frames = [env.render()]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Greedy policy: select the best learned action
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

    env.close()

    # Save animation
    imageio.mimsave(output_path, frames, duration=500, loop=0)

    print("GIF saved:", output_path)
    print("Goal reached:", bool(reward == 1))


if __name__ == "__main__":
    record_agent()
