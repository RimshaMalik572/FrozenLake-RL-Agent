
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent


def main():
    # Reproducible results
    np.random.seed(42)

    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    print("Environment: FrozenLake-v1")
    print("States:", state_size)
    print("Actions:", action_size)

    # Create Q-Learning agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        min_epsilon=0.01,
    )

    episodes = 50000
    rewards_history = []
    recent_successes = []

    # Train the agent
    for episode in range(episodes):
        state, info = env.reset(seed=42 + episode)
        total_reward = 0

        while True:
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            agent.learn(
                state,
                action,
                reward,
                next_state,
                terminated or truncated,
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()

        rewards_history.append(total_reward)
        recent_successes.append(total_reward)

        if (episode + 1) % 1000 == 0:
            success_rate = np.mean(recent_successes[-1000:])
            print(
                f"Episode: {episode + 1}/{episodes} | "
                f"Success rate: {success_rate:.2%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save trained Q-table
    agent.save("q_table.npy")
    print("\nQ-table saved as q_table.npy")

    # Plot training progress
    window = 500
    rewards_array = np.array(rewards_history)

    moving_average = np.convolve(
        rewards_array,
        np.ones(window) / window,
        mode="valid",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average)
    plt.title("FrozenLake Q-Learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (500 episodes)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_rewards.png", dpi=150)
    plt.show(block=False)
    plt.close() 

    print("Training graph saved as training_rewards.png")

    # Evaluate the learned policy without exploration
    evaluation_episodes = 100
    successes = 0

    for episode in range(evaluation_episodes):
        state, info = env.reset(seed=100000 + episode)

        while True:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                if reward == 1:
                    successes += 1
                break

    evaluation_rate = successes / evaluation_episodes

    print("\nEvaluation Results")
    print("------------------")
    print("Evaluation episodes:", evaluation_episodes)
    print("Successful episodes:", successes)
    print(f"Success rate: {evaluation_rate:.2%}")

    env.close()


if __name__ == "__main__":
    main()
