import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== Environment ====================
class GridWorld:
    """Simple GridWorld environment for navigation"""
    def __init__(self, size=10):
        self.size = size
        self.goal = np.array([size-1, size-1])
        self.reset()
    
    def reset(self):
        """Reset agent to start position"""
        self.state = np.array([0, 0])
        return self.state.copy()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        new_state = self.state + moves[action]
        
        # Check boundaries
        new_state = np.clip(new_state, 0, self.size - 1)
        self.state = new_state
        
        # Reward function
        done = np.array_equal(self.state, self.goal)
        reward = 100.0 if done else -1.0
        
        return self.state.copy(), reward, done
    
    def get_state_vector(self):
        """Return normalized state vector"""
        return self.state / self.size


# ==================== Neural Networks ====================
class PolicyNetwork(nn.Module):
    """Policy network for selecting actions"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class ValueNetwork(nn.Module):
    """Value network for critic (Actor-Critic)"""
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# ==================== REINFORCE Algorithm ====================
class REINFORCE:
    """Monte Carlo Policy Gradient (REINFORCE)"""
    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode memory"""
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action and store log probability"""
        action, log_prob = self.policy.select_action(state)
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def update(self):
        """Update policy using episode returns"""
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns for stability
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()


# ==================== Actor-Critic Algorithm ====================
class ActorCritic:
    """Actor-Critic with TD learning"""
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 actor_lr=0.001, critic_lr=0.005, gamma=0.99):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
    
    def select_action(self, state):
        """Select action using current policy"""
        action, log_prob = self.actor.select_action(state)
        return action, log_prob
    
    def update(self, state, action, reward, next_state, done, log_prob):
        """Update actor and critic using TD error"""
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        
        # Compute TD target and error
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.FloatTensor([0])
        td_target = reward + self.gamma * next_value
        td_error = td_target - value
        
        # Update critic (value network)
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor (policy network)
        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()


# ==================== Training Functions ====================
def train_reinforce(env, agent, num_episodes=500):
    """Train using REINFORCE algorithm"""
    episode_rewards = []
    episode_losses = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()
        total_reward = 0
        steps = 0
        
        for step in range(200):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_reward(reward)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Update policy after episode
        loss = agent.update()
        
        episode_rewards.append(total_reward)
        episode_losses.append(loss)
        episode_lengths.append(steps)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Steps: {steps}")
    
    return episode_rewards, episode_losses, episode_lengths


def train_actor_critic(env, agent, num_episodes=500):
    """Train using Actor-Critic algorithm"""
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        actor_losses = []
        critic_losses = []
        
        for step in range(200):  # Max steps per episode
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update after each step
            actor_loss, critic_loss = agent.update(
                state, action, reward, next_state, done, log_prob
            )
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_actor_losses.append(np.mean(actor_losses))
        episode_critic_losses.append(np.mean(critic_losses))
        episode_lengths.append(steps)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Steps: {steps}")
    
    return episode_rewards, episode_actor_losses, episode_critic_losses, episode_lengths


# ==================== Visualization ====================
def plot_training_results(results, algorithm_name):
    """Plot training metrics"""
    if algorithm_name == "REINFORCE":
        rewards, losses, lengths = results
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:  # Actor-Critic
        rewards, actor_losses, critic_losses, lengths = results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Smooth rewards using moving average
    window = 20
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # Plot 1: Episode Rewards
    axes[0] if algorithm_name == "REINFORCE" else axes[0, 0]
    ax = axes[0] if algorithm_name == "REINFORCE" else axes[0, 0]
    ax.plot(rewards, alpha=0.3, label='Raw')
    ax.plot(range(window-1, len(rewards)), smoothed_rewards, label='Smoothed', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'{algorithm_name} - Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Losses
    if algorithm_name == "REINFORCE":
        ax = axes[1]
        ax.plot(losses)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
    else:
        ax = axes[0, 1]
        ax.plot(actor_losses, label='Actor Loss')
        ax.plot(critic_losses, label='Critic Loss', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    ax = axes[2] if algorithm_name == "REINFORCE" else axes[1, 0]
    ax.plot(lengths)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Goal')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Moving average reward (Actor-Critic only)
    if algorithm_name == "Actor-Critic":
        ax = axes[1, 1]
        moving_avg = [np.mean(rewards[max(0, i-50):i+1]) for i in range(len(rewards))]
        ax.plot(moving_avg, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward (50 episodes)')
        ax.set_title('Moving Average Reward')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{algorithm_name.lower()}_training.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_policy(env, agent, algorithm_name):
    """Visualize learned policy on GridWorld"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    state = env.reset()
    trajectory = [state.copy()]
    
    for _ in range(50):
        if algorithm_name == "REINFORCE":
            action = agent.select_action(state)
        else:  # Actor-Critic
            action, _ = agent.select_action(state)
        
        state, reward, done = env.step(action)
        trajectory.append(state.copy())
        
        if done:
            break
    
    trajectory = np.array(trajectory)
    
    # Draw grid
    for i in range(env.size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Draw goal
    goal_rect = Rectangle((env.goal[1], env.goal[0]), 1, 1, 
                          facecolor='green', alpha=0.5, label='Goal')
    ax.add_patch(goal_rect)
    
    # Draw start
    start_rect = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.3, label='Start')
    ax.add_patch(start_rect)
    
    # Draw trajectory
    ax.plot(trajectory[:, 1] + 0.5, trajectory[:, 0] + 0.5, 
            'ro-', linewidth=2, markersize=8, alpha=0.6, label='Trajectory')
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{algorithm_name} - Learned Policy Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{algorithm_name.lower()}_policy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{algorithm_name} Policy:")
    print(f"Steps to goal: {len(trajectory) - 1}")
    print(f"Final position: {trajectory[-1]}")


def compare_algorithms(reinforce_rewards, ac_rewards):
    """Compare REINFORCE and Actor-Critic performance"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    window = 20
    reinforce_smooth = np.convolve(reinforce_rewards, np.ones(window)/window, mode='valid')
    ac_smooth = np.convolve(ac_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(range(window-1, len(reinforce_rewards)), reinforce_smooth, 
            label='REINFORCE', linewidth=2)
    ax.plot(range(window-1, len(ac_rewards)), ac_smooth, 
            label='Actor-Critic', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (Smoothed)')
    ax.set_title('Algorithm Comparison: REINFORCE vs Actor-Critic')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== Main Execution ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Policy Gradient Methods Lab")
    print("=" * 60)
    
    # Environment and agent parameters
    env = GridWorld(size=10)
    state_dim = 2
    hidden_dim = 64
    action_dim = 4
    num_episodes = 500
    
    # ========== Train REINFORCE ==========
    print("\n[1/2] Training REINFORCE...")
    print("-" * 60)
    reinforce_agent = REINFORCE(state_dim, hidden_dim, action_dim, lr=0.001)
    reinforce_results = train_reinforce(env, reinforce_agent, num_episodes)
    
    print("\n[REINFORCE] Training completed!")
    print(f"Final 50-episode average reward: {np.mean(reinforce_results[0][-50:]):.2f}")
    
    # ========== Train Actor-Critic ==========
    print("\n[2/2] Training Actor-Critic...")
    print("-" * 60)
    ac_agent = ActorCritic(state_dim, hidden_dim, action_dim, 
                          actor_lr=0.001, critic_lr=0.005)
    ac_results = train_actor_critic(env, ac_agent, num_episodes)
    
    print("\n[Actor-Critic] Training completed!")
    print(f"Final 50-episode average reward: {np.mean(ac_results[0][-50:]):.2f}")
    
    # ========== Visualizations ==========
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    # Plot training results
    plot_training_results(reinforce_results, "REINFORCE")
    plot_training_results(ac_results, "Actor-Critic")
    
    # Visualize learned policies
    visualize_policy(env, reinforce_agent, "REINFORCE")
    visualize_policy(env, ac_agent, "Actor-Critic")
    
    # Compare algorithms
    compare_algorithms(reinforce_results[0], ac_results[0])
    
    print("\n" + "=" * 60)
    print("Lab Complete! All visualizations saved.")
    print("=" * 60)