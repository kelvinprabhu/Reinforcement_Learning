import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
np.random.seed(42)

# ========================
# ENVIRONMENT
# ========================

class GridWorldEnvironment:
    """Enhanced GridWorld with multiple reward zones"""
    def __init__(self, size=6):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # Define special states
        self.obstacles = [(1, 1), (2, 2), (3, 3), (1, 3)]
        self.rewards = [(4, 1), (1, 4)]  # Bonus reward states
        self.penalties = [(2, 4), (4, 2)]  # Penalty states
        
        self.state = self.start
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_state = (self.state[0] + moves[action][0], 
                      self.state[1] + moves[action][1])
        
        # Boundary check
        if (next_state[0] < 0 or next_state[0] >= self.size or 
            next_state[1] < 0 or next_state[1] >= self.size):
            next_state = self.state
        
        # Obstacle check
        if next_state in self.obstacles:
            next_state = self.state
            reward = -1
            done = False
        elif next_state == self.goal:
            reward = 100
            done = True
        elif next_state in self.rewards:
            reward = 10
            done = False
        elif next_state in self.penalties:
            reward = -10
            done = False
        else:
            reward = -0.1
            done = False
            
        self.state = next_state
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


# ========================
# TD(0) ALGORITHM
# ========================

class TD0Agent:
    """TD(0) Agent with tracking for visualization"""
    def __init__(self, env, alpha=0.1, gamma=0.95):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.V = defaultdict(float)  # State values
        
        # Tracking for visualization
        self.episode_history = []
        self.value_history = []
        self.td_errors = []
        self.state_visits = defaultdict(int)
        
    def random_policy(self, state):
        """Simple random policy"""
        return np.random.choice(self.env.get_actions())
    
    def train_episode(self):
        """Train for one episode and track data"""
        state = self.env.reset()
        episode_data = {
            'states': [state],
            'rewards': [],
            'td_errors': [],
            'values_before': [],
            'values_after': []
        }
        
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Store value before update
            episode_data['values_before'].append(self.V[state])
            
            # Take action
            action = self.random_policy(state)
            next_state, reward, done = self.env.step(action)
            
            # TD(0) Update
            td_target = reward + self.gamma * self.V[next_state]
            td_error = td_target - self.V[state]
            self.V[state] += self.alpha * td_error
            
            # Track data
            episode_data['states'].append(next_state)
            episode_data['rewards'].append(reward)
            episode_data['td_errors'].append(td_error)
            episode_data['values_after'].append(self.V[state])
            self.state_visits[state] += 1
            
            state = next_state
            steps += 1
        
        return episode_data
    
    def train(self, episodes=200):
        """Train for multiple episodes"""
        episode_returns = []
        episode_lengths = []
        
        for episode in range(episodes):
            episode_data = self.train_episode()
            
            # Calculate return
            total_return = sum(episode_data['rewards'])
            episode_returns.append(total_return)
            episode_lengths.append(len(episode_data['rewards']))
            
            # Store for visualization
            if episode % 10 == 0:
                self.episode_history.append(episode_data)
                self.value_history.append(dict(self.V))
            
            # Store TD errors
            if episode_data['td_errors']:
                self.td_errors.extend(episode_data['td_errors'])
        
        return episode_returns, episode_lengths
    
    def get_value_grid(self):
        """Get values as a grid for visualization"""
        grid = np.zeros((self.env.size, self.env.size))
        for i in range(self.env.size):
            for j in range(self.env.size):
                grid[i, j] = self.V[(i, j)]
        return grid


# ========================
# VISUALIZATION FUNCTIONS
# ========================

def plot_environment(env, ax, title="GridWorld Environment"):
    """Visualize the environment layout"""
    grid = np.ones((env.size, env.size, 3))
    
    # Color coding
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.start:
                grid[i, j] = [0.2, 0.8, 0.2]  # Green for start
            elif (i, j) == env.goal:
                grid[i, j] = [1.0, 0.84, 0.0]  # Gold for goal
            elif (i, j) in env.obstacles:
                grid[i, j] = [0.3, 0.3, 0.3]  # Gray for obstacles
            elif (i, j) in env.rewards:
                grid[i, j] = [0.5, 0.8, 1.0]  # Light blue for rewards
            elif (i, j) in env.penalties:
                grid[i, j] = [1.0, 0.5, 0.5]  # Light red for penalties
            else:
                grid[i, j] = [0.95, 0.95, 0.95]  # White for normal
    
    ax.imshow(grid, interpolation='nearest')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='Start'),
        mpatches.Patch(color=[1.0, 0.84, 0.0], label='Goal (+100)'),
        mpatches.Patch(color=[0.5, 0.8, 1.0], label='Reward (+10)'),
        mpatches.Patch(color=[1.0, 0.5, 0.5], label='Penalty (-10)'),
        mpatches.Patch(color=[0.3, 0.3, 0.3], label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)


def plot_value_function(agent, env, ax, title="Learned State Values", show_numbers=True):
    """Visualize the learned value function"""
    value_grid = agent.get_value_grid()
    
    # Create custom colormap
    im = ax.imshow(value_grid, cmap='RdYlGn', interpolation='nearest')
    
    # Add value numbers
    if show_numbers:
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) not in env.obstacles:
                    text = ax.text(j, i, f'{value_grid[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, color='white', linewidth=0.5)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_trajectory(agent, env, ax, episode_data, title="Agent Trajectory"):
    """Visualize agent's trajectory through the environment"""
    # Plot environment as background
    grid = np.ones((env.size, env.size, 3)) * 0.95
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                grid[i, j] = [0.3, 0.3, 0.3]
    
    ax.imshow(grid, interpolation='nearest')
    
    # Plot trajectory
    states = episode_data['states']
    if states:
        y_coords = [s[0] for s in states]
        x_coords = [s[1] for s in states]
        
        # Plot path with gradient
        for i in range(len(states)-1):
            alpha = 0.3 + 0.7 * (i / len(states))
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   'b-', linewidth=2, alpha=alpha)
        
        # Mark start and end
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, label='End')
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    ax.legend(loc='upper right')


def plot_learning_curves(returns, lengths, td_errors, axes):
    """Plot various learning metrics"""
    
    # Smooth function
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0] if data else 0
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # Plot 1: Episode Returns
    ax1 = axes[0]
    episodes = range(len(returns))
    ax1.plot(episodes, returns, alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth(returns), linewidth=2, color='darkblue', label='Smoothed')
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Total Return', fontsize=10)
    ax1.set_title('Learning Progress: Returns per Episode', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2 = axes[1]
    ax2.plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
    ax2.plot(episodes, smooth(lengths), linewidth=2, color='darkgreen', label='Smoothed')
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Steps to Goal', fontsize=10)
    ax2.set_title('Learning Progress: Episode Length', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TD Error Distribution
    ax3 = axes[2]
    ax3.hist(td_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('TD Error', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('TD Error Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


def plot_value_evolution(value_history, env, axes):
    """Show how values evolve over training"""
    snapshots = [0, len(value_history)//3, 2*len(value_history)//3, -1]
    titles = ['Early Training', 'Mid Training', 'Late Training', 'Final Values']
    
    for idx, (snapshot, title) in enumerate(zip(snapshots, titles)):
        ax = axes[idx]
        values = value_history[snapshot]
        
        # Create value grid
        grid = np.zeros((env.size, env.size))
        for i in range(env.size):
            for j in range(env.size):
                grid[i, j] = values.get((i, j), 0)
        
        im = ax.imshow(grid, cmap='RdYlGn', interpolation='nearest')
        
        # Add numbers
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) not in env.obstacles:
                    ax.text(j, i, f'{grid[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=7)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.grid(True, color='white', linewidth=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046)


# ========================
# MAIN EXECUTION
# ========================

def run_td0_experiment():
    """Run complete TD(0) experiment with visualizations"""
    
    print("="*70)
    print("TD(0) TEMPORAL DIFFERENCE LEARNING - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Create environment
    env = GridWorldEnvironment(size=6)
    
    # Create and train agent
    print("\n[1] Initializing TD(0) Agent...")
    print(f"    Learning Rate (α): 0.1")
    print(f"    Discount Factor (γ): 0.95")
    
    agent = TD0Agent(env, alpha=0.1, gamma=0.95)
    
    print("\n[2] Training agent for 200 episodes...")
    returns, lengths = agent.train(episodes=200)
    
    print(f"\n[3] Training Complete!")
    print(f"    Average Return (last 50 episodes): {np.mean(returns[-50:]):.2f}")
    print(f"    Average Length (last 50 episodes): {np.mean(lengths[-50:]):.2f}")
    print(f"    Total TD Errors Recorded: {len(agent.td_errors)}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.4)
    
    # Row 1: Environment and Final Values
    ax1 = fig.add_subplot(gs[0, :2])
    plot_environment(env, ax1, "GridWorld Environment Layout")
    
    ax2 = fig.add_subplot(gs[0, 2:])
    plot_value_function(agent, env, ax2, "Final Learned State Values")
    
    # Row 2: Learning Curves
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :2])
    ax5 = fig.add_subplot(gs[2, 2:])
    plot_learning_curves(returns, lengths, agent.td_errors, [ax3, ax4, ax5])
    
    # Row 3: Value Evolution
    if len(agent.value_history) >= 4:
        axes_evolution = [fig.add_subplot(gs[3, i]) for i in range(4)]
        plot_value_evolution(agent.value_history, env, axes_evolution)
    
    plt.suptitle('TD(0) Algorithm: Complete Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('td0_analysis.png', dpi=150, bbox_inches='tight')
    print("\n[4] Visualization saved as 'td0_analysis.png'")
    plt.show()
    
    # Show sample trajectory
    print("\n[5] Generating sample trajectory visualization...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run a test episode
    test_data = agent.train_episode()
    
    plot_trajectory(agent, env, axes2[0], test_data, "Sample Episode Trajectory")
    plot_environment(env, axes2[1], "Environment")
    plot_value_function(agent, env, axes2[2], "Current Value Function")
    
    plt.tight_layout()
    plt.savefig('td0_trajectory.png', dpi=150, bbox_inches='tight')
    print("    Trajectory visualization saved as 'td0_trajectory.png'")
    plt.show()
    
    # Print TD(0) Algorithm Details
    print("\n" + "="*70)
    print("TD(0) ALGORITHM EXPLANATION")
    print("="*70)
    print("""
TD(0) Update Rule:
    V(s) ← V(s) + α[R + γV(s') - V(s)]
    
Where:
    • V(s)  = Value of current state
    • α     = Learning rate (0.1)
    • R     = Immediate reward
    • γ     = Discount factor (0.95)
    • V(s') = Value of next state
    
Key Characteristics:
    1. Bootstrap: Uses estimate V(s') to update V(s)
    2. Online: Updates after each step
    3. Bias-Variance: Lower variance than MC, some bias
    4. Sample Efficient: Learns from incomplete episodes
    
TD Error:
    δ = R + γV(s') - V(s)
    • Positive δ: State value should increase
    • Negative δ: State value should decrease
    • Zero δ: Prediction is accurate
    """)
    
    print("="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    run_td0_experiment()