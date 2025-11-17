import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import seaborn as sns
from IPython.display import HTML

# Set style
sns.set_style("whitegrid")
np.random.seed(42)

# ========================
# ENVIRONMENT
# ========================

class GridWorldEnvironment:
    """Enhanced GridWorld for SARSA demonstration"""
    def __init__(self, size=8):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # Define zones
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2),
                         (5, 5), (5, 6), (6, 5)]
        self.high_rewards = [(3, 6), (6, 3)]
        self.penalties = [(1, 5), (5, 1), (4, 4)]
        
        self.state = self.start
        self.action_names = ['↑', '→', '↓', '←']
        
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
            reward = -1
            done = False
        # Obstacle check
        elif next_state in self.obstacles:
            next_state = self.state
            reward = -2
            done = False
        # Goal
        elif next_state == self.goal:
            reward = 100
            done = True
        # High rewards
        elif next_state in self.high_rewards:
            reward = 20
            done = False
        # Penalties
        elif next_state in self.penalties:
            reward = -15
            done = False
        else:
            reward = -0.1
            done = False
            
        self.state = next_state
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


# ========================
# SARSA AGENT
# ========================

class SARSAAgent:
    """SARSA Agent with detailed tracking"""
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.Q = defaultdict(lambda: np.zeros(4))
        
        # Tracking
        self.episode_history = []
        self.q_value_history = []
        self.policy_history = []
        self.decision_log = []
        
    def epsilon_greedy_policy(self, state, explore=True):
        """Epsilon-greedy action selection with tracking"""
        if explore and np.random.random() < self.epsilon:
            action = np.random.choice(self.env.get_actions())
            decision_type = "explore"
        else:
            action = np.argmax(self.Q[state])
            decision_type = "exploit"
        
        return action, decision_type
    
    def get_action_probs(self, state):
        """Get probability distribution over actions"""
        q_values = self.Q[state]
        best_action = np.argmax(q_values)
        probs = np.ones(4) * (self.epsilon / 4)
        probs[best_action] += (1 - self.epsilon)
        return probs
    
    def train_episode(self, record_details=False):
        """Train one episode with detailed tracking"""
        state = self.env.reset()
        action, decision_type = self.epsilon_greedy_policy(state)
        
        episode_data = {
            'states': [state],
            'actions': [action],
            'rewards': [],
            'q_values': [],
            'td_errors': [],
            'decision_types': [decision_type],
            'action_probs': []
        }
        
        done = False
        steps = 0
        max_steps = 150
        
        while not done and steps < max_steps:
            # Store Q-values before update
            q_before = self.Q[state].copy()
            episode_data['q_values'].append(q_before)
            episode_data['action_probs'].append(self.get_action_probs(state))
            
            # Take action
            next_state, reward, done = self.env.step(action)
            next_action, next_decision_type = self.epsilon_greedy_policy(next_state)
            
            # SARSA Update
            td_target = reward + self.gamma * self.Q[next_state][next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            # Track
            episode_data['states'].append(next_state)
            episode_data['actions'].append(next_action)
            episode_data['rewards'].append(reward)
            episode_data['td_errors'].append(td_error)
            episode_data['decision_types'].append(next_decision_type)
            
            # Decision log for detailed view
            if record_details and steps < 30:
                self.decision_log.append({
                    'step': steps,
                    'state': state,
                    'action': action,
                    'action_name': self.env.action_names[action],
                    'decision': decision_type,
                    'q_values': q_before,
                    'reward': reward,
                    'td_error': td_error,
                    'next_state': next_state
                })
            
            state = next_state
            action = next_action
            decision_type = next_decision_type
            steps += 1
        
        return episode_data
    
    def train(self, episodes=300):
        """Train for multiple episodes"""
        episode_returns = []
        episode_lengths = []
        
        for episode in range(episodes):
            record = (episode % 30 == 0) or episode < 5 or episode > episodes - 5
            episode_data = self.train_episode(record_details=record)
            
            total_return = sum(episode_data['rewards'])
            episode_returns.append(total_return)
            episode_lengths.append(len(episode_data['rewards']))
            
            if record:
                self.episode_history.append(episode_data)
                self.q_value_history.append(dict(self.Q))
                self.policy_history.append(self.get_policy())
        
        return episode_returns, episode_lengths
    
    def get_policy(self):
        """Extract greedy policy from Q-values"""
        policy = {}
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                policy[state] = np.argmax(self.Q[state])
        return policy


# ========================
# VISUALIZATION FUNCTIONS
# ========================

def plot_environment(env, ax, title="Environment"):
    """Plot environment with all zones"""
    grid = np.ones((env.size, env.size, 3))
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.start:
                grid[i, j] = [0.2, 0.8, 0.2]  # Green
            elif (i, j) == env.goal:
                grid[i, j] = [1.0, 0.84, 0.0]  # Gold
            elif (i, j) in env.obstacles:
                grid[i, j] = [0.2, 0.2, 0.2]  # Dark gray
            elif (i, j) in env.high_rewards:
                grid[i, j] = [0.4, 0.7, 1.0]  # Light blue
            elif (i, j) in env.penalties:
                grid[i, j] = [1.0, 0.4, 0.4]  # Light red
            else:
                grid[i, j] = [0.95, 0.95, 0.95]
    
    ax.imshow(grid, interpolation='nearest')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='Start'),
        mpatches.Patch(color=[1.0, 0.84, 0.0], label='Goal (+100)'),
        mpatches.Patch(color=[0.4, 0.7, 1.0], label='High Reward (+20)'),
        mpatches.Patch(color=[1.0, 0.4, 0.4], label='Penalty (-15)'),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='center left', 
             bbox_to_anchor=(1.02, 0.5), fontsize=8)


def plot_agent_trajectory_animated(env, episode_data, ax, title="Agent Movement"):
    """Show agent's path with decision indicators"""
    # Background
    grid = np.ones((env.size, env.size, 3)) * 0.95
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                grid[i, j] = [0.3, 0.3, 0.3]
            elif (i, j) == env.goal:
                grid[i, j] = [1.0, 0.9, 0.0]
    
    ax.imshow(grid, interpolation='nearest')
    
    states = episode_data['states']
    actions = episode_data['actions']
    decisions = episode_data['decision_types']
    
    # Plot path with colors for explore/exploit
    for i in range(len(states)-1):
        y1, x1 = states[i]
        y2, x2 = states[i+1]
        
        color = 'orange' if decisions[i] == 'explore' else 'blue'
        alpha = 0.3 + 0.5 * (i / len(states))
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, 
                                 color=color, alpha=alpha))
        
        # Mark decision type
        marker = 'o' if decisions[i] == 'explore' else 's'
        ax.plot(x1, y1, marker, color=color, markersize=6, alpha=alpha)
    
    # Start and end markers
    ax.plot(states[0][1], states[0][0], 'go', markersize=15, 
           label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(states[-1][1], states[-1][0], 'r*', markersize=20, 
           label='End', markeredgecolor='darkred', markeredgewidth=2)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(color='blue', label='Exploit (Greedy)'),
        mpatches.Patch(color='orange', label='Explore (Random)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='g', markersize=10, label='Start'),
        plt.Line2D([0], [0], marker='*', color='w', 
                  markerfacecolor='r', markersize=12, label='End')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def plot_policy_arrows(agent, env, ax, title="Learned Policy (Action Selection)"):
    """Visualize policy with arrows"""
    grid = np.ones((env.size, env.size, 3)) * 0.95
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                grid[i, j] = [0.3, 0.3, 0.3]
    
    ax.imshow(grid, interpolation='nearest', alpha=0.3)
    
    # Arrow directions
    arrow_dirs = [(-0.35, 0), (0, 0.35), (0.35, 0), (0, -0.35)]
    
    policy = agent.get_policy()
    
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state not in env.obstacles and state != env.goal:
                action = policy[state]
                q_values = agent.Q[state]
                
                # Color based on Q-value
                max_q = np.max(q_values)
                if max_q > 50:
                    color = 'darkgreen'
                elif max_q > 10:
                    color = 'green'
                elif max_q > 0:
                    color = 'blue'
                else:
                    color = 'red'
                
                dy, dx = arrow_dirs[action]
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.15,
                        fc=color, ec=color, linewidth=2, alpha=0.8)
    
    # Mark goal
    ax.plot(env.goal[1], env.goal[0], 'y*', markersize=25, 
           markeredgecolor='orange', markeredgewidth=2)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)


def plot_q_value_heatmap(agent, env, ax, action_idx, action_name):
    """Plot Q-values for specific action"""
    grid = np.zeros((env.size, env.size))
    
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            grid[i, j] = agent.Q[state][action_idx]
    
    im = ax.imshow(grid, cmap='RdYlGn', interpolation='nearest')
    
    # Add values
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) not in env.obstacles:
                ax.text(j, i, f'{grid[i, j]:.1f}',
                       ha="center", va="center", 
                       color="black", fontsize=7, fontweight='bold')
    
    ax.set_title(f'Q-Values: Action {action_name}', 
                fontsize=10, fontweight='bold')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True, color='white', linewidth=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046)


def plot_decision_making_details(agent, ax):
    """Show detailed decision-making process"""
    if not agent.decision_log:
        ax.text(0.5, 0.5, 'No decision log available', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Take first 10 decisions
    decisions = agent.decision_log[:min(10, len(agent.decision_log))]
    
    data = []
    labels = []
    colors = []
    
    for i, dec in enumerate(decisions):
        labels.append(f"Step {dec['step']}\n{dec['action_name']}")
        data.append(dec['q_values'])
        colors.append('orange' if dec['decision'] == 'explore' else 'blue')
    
    # Create grouped bar chart
    x = np.arange(len(labels))
    width = 0.2
    action_names = ['↑', '→', '↓', '←']
    
    for j in range(4):
        values = [d[j] for d in data]
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=action_names[j], alpha=0.8)
    
    ax.set_xlabel('Decision Step', fontsize=9)
    ax.set_ylabel('Q-Value', fontsize=9)
    ax.set_title('Decision Making: Q-Values at Each Step', 
                fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(title='Actions', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)


def plot_learning_metrics(returns, lengths, ax1, ax2):
    """Plot learning progress"""
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0] if data else 0
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    episodes = range(len(returns))
    
    # Returns
    ax1.plot(episodes, returns, alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth(returns), linewidth=2.5, 
            color='darkblue', label='Smoothed')
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Total Return', fontsize=10)
    ax1.set_title('Learning Progress: Returns', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lengths
    ax2.plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
    ax2.plot(episodes, smooth(lengths), linewidth=2.5, 
            color='darkgreen', label='Smoothed')
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Episode Length', fontsize=10)
    ax2.set_title('Learning Progress: Steps to Goal', 
                 fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


def create_decision_table(agent, num_decisions=10):
    """Create a text table showing decision-making process"""
    if not agent.decision_log:
        return "No decision log available"
    
    table = "="*90 + "\n"
    table += "DETAILED DECISION-MAKING LOG\n"
    table += "="*90 + "\n"
    table += f"{'Step':<6} {'State':<10} {'Action':<8} {'Type':<10} {'Q-Values':<35} {'Reward':<8} {'TD-Error':<10}\n"
    table += "-"*90 + "\n"
    
    for dec in agent.decision_log[:num_decisions]:
        q_str = "[" + ", ".join([f"{q:.2f}" for q in dec['q_values']]) + "]"
        table += f"{dec['step']:<6} {str(dec['state']):<10} {dec['action_name']:<8} "
        table += f"{dec['decision']:<10} {q_str:<35} "
        table += f"{dec['reward']:<8.2f} {dec['td_error']:<10.3f}\n"
    
    table += "="*90 + "\n"
    return table


# ========================
# MAIN EXECUTION
# ========================

def run_sarsa_experiment():
    """Run complete SARSA experiment"""
    
    print("="*90)
    print("SARSA ALGORITHM: ACTION-VALUE ESTIMATION & POLICY IMPROVEMENT")
    print("="*90)
    
    # Create environment and agent
    env = GridWorldEnvironment(size=8)
    agent = SARSAAgent(env, alpha=0.1, gamma=0.95, epsilon=0.15)
    
    print("\n[1] Configuration:")
    print(f"    Environment Size: {env.size}x{env.size}")
    print(f"    Learning Rate (α): {agent.alpha}")
    print(f"    Discount Factor (γ): {agent.gamma}")
    print(f"    Exploration Rate (ε): {agent.epsilon}")
    
    print("\n[2] Training SARSA agent...")
    returns, lengths = agent.train(episodes=300)
    
    print(f"\n[3] Training Complete!")
    print(f"    Final Average Return: {np.mean(returns[-50:]):.2f}")
    print(f"    Final Average Steps: {np.mean(lengths[-50:]):.2f}")
    print(f"    Decisions Logged: {len(agent.decision_log)}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.4)
    
    # Row 1: Environment and Trajectories
    ax1 = fig.add_subplot(gs[0, :2])
    plot_environment(env, ax1, "Environment Layout")
    
    ax2 = fig.add_subplot(gs[0, 2:4])
    if agent.episode_history:
        plot_agent_trajectory_animated(env, agent.episode_history[-1], ax2, 
                                      "Final Episode: Agent Movement")
    
    ax3 = fig.add_subplot(gs[0, 4])
    plot_policy_arrows(agent, env, ax3, "Learned Policy")
    
    # Row 2: Q-Value Heatmaps
    action_names = ['Up ↑', 'Right →', 'Down ↓', 'Left ←']
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        plot_q_value_heatmap(agent, env, ax, i, action_names[i])
    
    # Decision making detail
    ax_dec = fig.add_subplot(gs[1, 4])
    plot_decision_making_details(agent, ax_dec)
    
    # Row 3: Learning Curves
    ax5 = fig.add_subplot(gs[2, :3])
    ax6 = fig.add_subplot(gs[2, 3:])
    plot_learning_metrics(returns, lengths, ax5, ax6)
    
    # Row 4: Episode comparisons
    if len(agent.episode_history) >= 3:
        episodes_to_show = [0, len(agent.episode_history)//2, -1]
        titles = ['Early Episode', 'Mid Training', 'Final Episode']
        
        for idx, (ep_idx, title) in enumerate(zip(episodes_to_show, titles)):
            ax = fig.add_subplot(gs[3, idx])
            plot_agent_trajectory_animated(env, agent.episode_history[ep_idx], 
                                         ax, title)
    
    plt.suptitle('SARSA Algorithm: Complete Agent Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('sarsa_complete_analysis.png', dpi=150, bbox_inches='tight')
    print("\n[4] Main visualization saved as 'sarsa_complete_analysis.png'")
    plt.show()
    
    # Print decision table
    print("\n" + create_decision_table(agent, num_decisions=15))
    
    # SARSA Algorithm Explanation
    print("\n" + "="*90)
    print("SARSA ALGORITHM EXPLANATION")
    print("="*90)
    print("""
SARSA Update Rule:
    Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
    
Where:
    • Q(s,a)  = Action-value for state s, action a
    • α       = Learning rate (0.1)
    • R       = Immediate reward
    • γ       = Discount factor (0.95)
    • Q(s',a')= Action-value for next state s', next action a'
    
Key Characteristics:
    1. ON-POLICY: Learns value of policy being followed
    2. BOOTSTRAPPING: Uses estimate Q(s',a') to update Q(s,a)
    3. ε-GREEDY: Balances exploration and exploitation
    4. SAFER: More conservative than Q-Learning
    
Decision Making:
    • EXPLOIT (1-ε = 0.85): Choose action with highest Q-value
    • EXPLORE (ε = 0.15): Choose random action
    
Action Selection:
    - Blue arrows: Greedy decisions (exploiting knowledge)
    - Orange arrows: Random decisions (exploring environment)
    
Policy Visualization:
    - Arrows show best action in each state
    - Color indicates Q-value magnitude:
        * Dark Green: High value (>50)
        * Green: Good value (10-50)
        * Blue: Positive value (0-10)
        * Red: Negative value (<0)
    """)
    
    print("="*90)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*90)


if __name__ == "__main__":
    run_sarsa_experiment()
    