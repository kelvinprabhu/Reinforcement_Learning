import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions: up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        self.reasoning = ""
        
    def get_state_key(self, state):
        return f"{state[0]},{state[1]}"
    
    def choose_action(self, state, explore=True):
        state_key = self.get_state_key(state)
        
        # Epsilon-greedy policy
        if explore and np.random.random() < self.epsilon:
            action_idx = np.random.randint(4)
            self.reasoning = f"🎲 Exploring: Random action '{self.actions[action_idx]}'"
            return action_idx
        
        q_values = self.q_table[state_key]
        action_idx = np.argmax(q_values)
        self.reasoning = f"🧠 Exploiting: Q-values {q_values.round(2)} → '{self.actions[action_idx]}'"
        return action_idx
    
    def get_next_state(self, state, action_idx):
        action = self.actions[action_idx]
        delta = self.action_map[action]
        new_state = (state[0] + delta[0], state[1] + delta[1])
        
        # Check boundaries
        if (0 <= new_state[0] < self.env.size and 
            0 <= new_state[1] < self.env.size):
            return new_state
        return state
    
    def update_q_value(self, state, action_idx, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-Learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
    
    def get_best_action(self, state):
        state_key = self.get_state_key(state)
        q_values = self.q_table[state_key]
        if np.all(q_values == 0):
            return None
        return self.actions[np.argmax(q_values)]
    
    def train_episode(self, max_steps=50):
        state = self.env.start
        episode_reward = 0
        steps = 0
        trajectory = [state]
        
        while state != self.env.goal and steps < max_steps:
            action_idx = self.choose_action(state)
            next_state = self.get_next_state(state, action_idx)
            reward = self.env.get_reward(next_state)
            
            self.update_q_value(state, action_idx, reward, next_state)
            
            episode_reward += reward
            state = next_state
            trajectory.append(state)
            steps += 1
            
            # Stop if hit obstacle
            if self.env.is_obstacle(state):
                break
        
        return episode_reward, trajectory


class Environment:
    def __init__(self, env_type='grid'):
        self.environments = {
            'grid': {
                'name': 'Simple Grid World',
                'size': 5,
                'start': (0, 0),
                'goal': (4, 4),
                'obstacles': [(1, 1), (2, 2), (3, 1)],
                'rewards': {'goal': 100, 'obstacle': -50, 'step': -1}
            },
            'maze': {
                'name': 'Maze Navigation',
                'size': 6,
                'start': (0, 0),
                'goal': (5, 5),
                'obstacles': [(1, 0), (1, 1), (1, 2), (3, 2), (3, 3), (3, 4), (0, 3), (2, 4)],
                'rewards': {'goal': 100, 'obstacle': -30, 'step': -1}
            },
            'cliff': {
                'name': 'Cliff Walking',
                'size': 4,
                'start': (0, 0),
                'goal': (3, 3),
                'obstacles': [(3, 1), (3, 2)],
                'rewards': {'goal': 100, 'obstacle': -100, 'step': -1}
            }
        }
        
        config = self.environments[env_type]
        self.name = config['name']
        self.size = config['size']
        self.start = config['start']
        self.goal = config['goal']
        self.obstacles = config['obstacles']
        self.rewards = config['rewards']
    
    def is_obstacle(self, state):
        return state in self.obstacles
    
    def is_goal(self, state):
        return state == self.goal
    
    def get_reward(self, state):
        if self.is_goal(state):
            return self.rewards['goal']
        if self.is_obstacle(state):
            return self.rewards['obstacle']
        return self.rewards['step']


class QLearningVisualizer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 12))
        self.fig.suptitle(f'Q-Learning: {env.name}', fontsize=16, fontweight='bold')
        
        # Initialize plots
        self.ax_env = self.axes[0, 0]
        self.ax_policy = self.axes[0, 1]
        self.ax_reward = self.axes[1, 0]
        self.ax_stats = self.axes[1, 1]
        
        self.reward_history = []
        self.episode_count = 0
        
        plt.tight_layout()
    
    def draw_grid(self, ax, title, show_policy=False, agent_pos=None):
        ax.clear()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw grid
        for i in range(self.env.size + 1):
            ax.axhline(i, color='gray', linewidth=0.5)
            ax.axvline(i, color='gray', linewidth=0.5)
        
        # Draw obstacles
        for obs in self.env.obstacles:
            rect = patches.Rectangle((obs[1], obs[0]), 1, 1, 
                                    linewidth=2, edgecolor='red', 
                                    facecolor='red', alpha=0.6)
            ax.add_patch(rect)
            ax.text(obs[1] + 0.5, obs[0] + 0.5, '⛔', 
                   ha='center', va='center', fontsize=20)
        
        # Draw goal
        goal_rect = patches.Rectangle((self.env.goal[1], self.env.goal[0]), 1, 1,
                                     linewidth=2, edgecolor='green',
                                     facecolor='green', alpha=0.4)
        ax.add_patch(goal_rect)
        ax.text(self.env.goal[1] + 0.5, self.env.goal[0] + 0.5, '🎯',
               ha='center', va='center', fontsize=20)
        
        # Draw agent
        if agent_pos:
            agent_rect = patches.Rectangle((agent_pos[1], agent_pos[0]), 1, 1,
                                          linewidth=3, edgecolor='yellow',
                                          facecolor='yellow', alpha=0.7)
            ax.add_patch(agent_rect)
            ax.text(agent_pos[1] + 0.5, agent_pos[0] + 0.5, '🤖',
                   ha='center', va='center', fontsize=20)
        
        # Draw policy arrows
        if show_policy:
            arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if (i, j) not in self.env.obstacles and (i, j) != self.env.goal:
                        best_action = self.agent.get_best_action((i, j))
                        if best_action:
                            ax.text(j + 0.5, i + 0.5, arrow_map[best_action],
                                   ha='center', va='center', fontsize=16,
                                   color='blue', fontweight='bold')
        
        ax.set_xticks(range(self.env.size + 1))
        ax.set_yticks(range(self.env.size + 1))
        ax.grid(True)
    
    def update_reward_plot(self):
        self.ax_reward.clear()
        self.ax_reward.set_title('Reward History', fontsize=12, fontweight='bold')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Total Reward')
        
        if self.reward_history:
            episodes = range(1, len(self.reward_history) + 1)
            self.ax_reward.plot(episodes, self.reward_history, 
                              'b-', linewidth=2, label='Episode Reward')
            
            # Moving average
            if len(self.reward_history) >= 10:
                window = 10
                moving_avg = np.convolve(self.reward_history, 
                                        np.ones(window)/window, mode='valid')
                self.ax_reward.plot(range(window, len(self.reward_history) + 1), 
                                   moving_avg, 'r-', linewidth=2, 
                                   label=f'{window}-Episode Moving Avg')
            
            self.ax_reward.legend()
            self.ax_reward.grid(True, alpha=0.3)
    
    def update_stats(self):
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        last_reward = self.reward_history[-1] if self.reward_history else 0
        best_reward = max(self.reward_history) if self.reward_history else 0
        worst_reward = min(self.reward_history) if self.reward_history else 0
        
        # Handle average reward display
        if len(self.reward_history) >= 10:
            avg_reward_str = f"{np.mean(self.reward_history[-10:]):.1f}"
        else:
            avg_reward_str = "N/A"
        
        stats_text = f"""
        STATISTICS
        {'='*40}
        
        Episode: {self.episode_count}
        
        Last Reward: {last_reward:.1f}
        
        Avg Reward (Last 10): {avg_reward_str}
        
        Best Reward: {best_reward:.1f}
        
        Worst Reward: {worst_reward:.1f}
        
        {'='*40}
        AGENT REASONING
        
        {self.agent.reasoning}
        
        {'='*40}
        PARAMETERS
        
        Learning Rate (α): {self.agent.alpha}
        Discount Factor (γ): {self.agent.gamma}
        Exploration Rate (ε): {self.agent.epsilon}
        """
        
        self.ax_stats.text(0.1, 0.9, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def visualize_episode(self, trajectory):
        for i, pos in enumerate(trajectory):
            self.draw_grid(self.ax_env, f'Environment - Step {i+1}', agent_pos=pos)
            self.draw_grid(self.ax_policy, 'Learned Policy Map', show_policy=True)
            plt.pause(0.1)
    
    def train_and_visualize(self, num_episodes=100):
        for episode in range(num_episodes):
            self.episode_count = episode + 1
            reward, trajectory = self.agent.train_episode()
            self.reward_history.append(reward)
            
            # Update visualizations
            if episode % 5 == 0:  # Update every 5 episodes for performance
                self.draw_grid(self.ax_env, 'Environment', agent_pos=trajectory[-1])
                self.draw_grid(self.ax_policy, 'Learned Policy Map', show_policy=True)
                self.update_reward_plot()
                self.update_stats()
                plt.pause(0.01)
            
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward:.1f}")
        
        # Final update
        self.draw_grid(self.ax_env, 'Environment (Final)', agent_pos=self.env.goal)
        self.draw_grid(self.ax_policy, 'Learned Policy Map (Final)', show_policy=True)
        self.update_reward_plot()
        self.update_stats()
        plt.show()


def main():
    print("=" * 60)
    print("Q-LEARNING REINFORCEMENT LEARNING")
    print("=" * 60)
    print("\nAvailable Environments:")
    print("1. Simple Grid World (5x5)")
    print("2. Maze Navigation (6x6)")
    print("3. Cliff Walking (4x4)")
    
    choice = input("\nSelect environment (1-3) [default: 1]: ").strip()
    env_map = {'1': 'grid', '2': 'maze', '3': 'cliff', '': 'grid'}
    env_type = env_map.get(choice, 'grid')
    
    num_episodes = input("Number of training episodes [default: 100]: ").strip()
    num_episodes = int(num_episodes) if num_episodes else 100
    
    print(f"\nInitializing {env_type} environment...")
    env = Environment(env_type)
    agent = QLearningAgent(env)
    visualizer = QLearningVisualizer(env, agent)
    
    print(f"Starting training for {num_episodes} episodes...")
    print("Close the plot window to exit.\n")
    
    visualizer.train_and_visualize(num_episodes)


if __name__ == "__main__":
    main()