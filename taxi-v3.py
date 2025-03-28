import gym
import numpy as np
import random
from collections import defaultdict

# Initialize Taxi-v3 environment
env = gym.make("Taxi-v3")


# =======================
# 1. Monte Carlo with Epsilon-Greedy
# =======================
def monte_carlo_epsilon_greedy(env, episodes=1000, epsilon=0.1, gamma=0.9):
    # Initialize Q(s, a) arbitrarily, here we initialize it to 0
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(episodes):
        # Generate an episode
        episode = []
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Update Q-value based on the episode
        G = 0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q[state][action] = returns_sum[state][action] / returns_count[state][action]

    return Q


# =======================
# 2. TD(0) with Q-learning
# =======================
def td0_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    # Initialize Q(s, a) arbitrarily
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state])  # Q-learning update
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            state = next_state

    return Q


# =======================
# 3. TD(lambda) with SARSA
# =======================
def td_lambda_sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, lambda_param=0.9, epsilon=0.1):
    # Initialize Q(s, a) arbitrarily
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    E = defaultdict(lambda: np.zeros(env.action_space.n))  # Eligibility traces

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)  # Choose action using epsilon-greedy
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)  # Choose next action

            # TD(λ) update rule for SARSA
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1  # Increment the eligibility trace

            # Update all Q(s, a) using eligibility traces
            for s in Q:
                for a in range(env.action_space.n):
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] = gamma * lambda_param * E[s][a]  # Decay eligibility traces

            state, action = next_state, next_action

    return Q


# =======================
# 4. TD(lambda) with Q-learning
# =======================
def td_lambda_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, lambda_param=0.9, epsilon=0.1):
    # Initialize Q(s, a) arbitrarily
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    E = defaultdict(lambda: np.zeros(env.action_space.n))  # Eligibility traces

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)  # Choose action using epsilon-greedy
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = np.argmax(Q[next_state])  # Choose the action that maximizes Q-value

            # TD(λ) update rule for Q-learning
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1  # Increment the eligibility trace

            # Update all Q(s, a) using eligibility traces
            for s in Q:
                for a in range(env.action_space.n):
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] = gamma * lambda_param * E[s][a]  # Decay eligibility traces

            state, action = next_state, next_action

    return Q


# =======================
# Helper function for epsilon-greedy strategy
# =======================
def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit


# =======================
# Run all the agents and evaluate performance
# =======================
def run_agents():
    # Define the number of episodes
    episodes = 1000

    # Run Monte Carlo with Epsilon-Greedy
    print("Training Monte Carlo with Epsilon-Greedy...")
    mc_q = monte_carlo_epsilon_greedy(env, episodes)

    # Run TD(0) with Q-learning
    print("Training TD(0) with Q-learning...")
    td_q = td0_q_learning(env, episodes)

    # Run TD(lambda) with SARSA
    print("Training TD(lambda) with SARSA...")
    td_lambda_sarsa_q = td_lambda_sarsa(env, episodes)

    # Run TD(lambda) with Q-learning
    print("Training TD(lambda) with Q-learning...")
    td_lambda_q_q = td_lambda_q_learning(env, episodes)

    # Displaying results or evaluation
    print("All agents trained successfully!")


# Run the agents
run_agents()