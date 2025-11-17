import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
np.random.seed(42)

# ========================
# ENVIRONMENTS
# ========================

class SimpleGridWorld:
    """Simple 5x5 GridWorld"""
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.state = self.start
        self.name = "Simple GridWorld"
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_state = (self.state[0] + moves[action][0], 
                      self.state[1] + moves[action][1])
        
        if (next_state[0] < 0 or next_state[0] >= self.size or 
            next_state[1] < 0 or next_state[1] >= self.size):
            next_state = self.state
        
        if next_state in self.obstacles:
            next_state = self.state
            
        self.state = next_state
        
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False
            
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


class WindyGridWorld:
    """GridWorld with wind that pushes agent upward"""
    def __init__(self):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        # Wind strength for each column
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.state = self.start
        self.name = "Windy GridWorld"
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Apply action
        next_state = (self.state[0] + moves[action][0], 
                      self.state[1] + moves[action][1])
        
        # Apply wind (pushes upward)
        wind_strength = self.wind[self.state[1]] if self.state[1] < len(self.wind) else 0
        next_state = (next_state[0] - wind_strength, next_state[1])
        
        # Boundary check
        next_state = (max(0, min(next_state[0], self.height-1)),
                      max(0, min(next_state[1], self.width-1)))
        
        self.state = next_state
        
        if self.state == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
            
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


class CliffWalkingWorld:
    """Cliff Walking - dangerous cliff area"""
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.state = self.start
        self.name = "Cliff Walking"
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_state = (self.state[0] + moves[action][0], 
                      self.state[1] + moves[action][1])
        
        # Boundary check
        next_state = (max(0, min(next_state[0], self.height-1)),
                      max(0, min(next_state[1], self.width-1)))
        
        # Check cliff
        if next_state in self.cliff:
            reward = -100
            done = False
            next_state = self.start
        elif next_state == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
            
        self.state = next_state
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


class MazeWorld:
    """Complex maze environment"""
    def __init__(self):
        self.size = 8
        self.start = (0, 0)
        self.goal = (7, 7)
        # Complex obstacle pattern
        self.obstacles = [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (3, 1), (3, 3), (3, 5), (3, 6),
            (5, 1), (5, 2), (5, 3), (5, 5), (5, 6),
            (2, 6), (4, 0), (6, 4)
        ]
        self.rewards = [(2, 4), (4, 2), (6, 6)]
        self.state = self.start
        self.name = "Complex Maze"
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_state = (self.state[0] + moves[action][0], 
                      self.state[1] + moves[action][1])
        
        if (next_state[0] < 0 or next_state[0] >= self.size or 
            next_state[1] < 0 or next_state[1] >= self.size):
            next_state = self.state
        
        if next_state in self.obstacles:
            next_state = self.state
            reward = -1
            done = False
        elif next_state == self.goal:
            reward = 100
            done = True
        elif next_state in self.rewards:
            reward = 5
            done = False
        else:
            reward = -0.1
            done = False
            
        self.state = next_state
        return self.state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]


# ========================
# TD(λ) AGENTS
# ========================

class TDLambdaAgent:
    """TD(λ) Agent with eligibility traces"""
    def __init__(self, env, alpha=0.1, gamma=0.95, lambda_=0.8):
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.lambda_ = lambda_      # Trace decay parameter
        self.Q = defaultdict(lambda: np.zeros(4))
        self.name = f"TD(λ={lambda_})"
        
        # Tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.trace_history = []
        
    def epsilon_greedy_policy(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.env.get_actions())
        else:
            return np.argmax(self.Q[state])
    
    def train_episode(self, epsilon=0.1):
        """Train one episode using TD(λ)"""
        state = self.env.reset()
        action = self.epsilon_greedy_policy(state, epsilon)
        
        # Initialize eligibility traces
        E = defaultdict(lambda: np.zeros(4))
        
        episode_data = {
            'states': [state],
            'actions': [action],
            'rewards': [],
            'traces': []
        }
        
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            next_state, reward, done = self.env.step(action)
            next_action = self.epsilon_greedy_policy(next_state, epsilon)
            
            # TD error
            td_target = reward + self.gamma * self.Q[next_state][next_action]
            td_error = td_target - self.Q[state][action]
            
            # Update eligibility trace for current state-action
            E[state][action] += 1
            
            # Store max trace for visualization
            max_trace = max([np.max(E[s]) for s in E.keys()])
            episode_data['traces'].append(max_trace)
            
            # Update all Q-values with traces
            for s in list(E.keys()):
                for a in range(4):
                    if E[s][a] > 0.01:  # Only update non-zero traces
                        self.Q[s][a] += self.alpha * td_error * E[s][a]
                        # Decay trace
                        E[s][a] *= self.gamma * self.lambda_
            
            episode_data['states'].append(next_state)
            episode_data['actions'].append(next_action)
            episode_data['rewards'].append(reward)
            
            state = next_state
            action = next_action
            steps += 1
        
        return episode_data
    
    def train(self, episodes=200, epsilon=0.1):
        """Train for multiple episodes"""
        for episode in range(episodes):
            episode_data = self.train_episode(epsilon)
            
            total_return = sum(episode_data['rewards'])
            self.episode_returns.append(total_return)
            self.episode_lengths.append(len(episode_data['rewards']))
            
            if episode % 20 == 0:
                self.trace_history.append(episode_data['traces'])
        
        return self.episode_returns, self.episode_lengths


class SARSA0Agent(TDLambdaAgent):
    """SARSA(0) for comparison - equivalent to TD(λ=0)"""
    def __init__(self, env, alpha=0.1, gamma=0.95):
        super().__init__(env, alpha, gamma, lambda_=0.0)
        self.name = "SARSA(0)"
    
    def train_episode(self, epsilon=0.1):
        """Standard SARSA without traces"""
        state = self.env.reset()
        action = self.epsilon_greedy_policy(state, epsilon)
        
        episode_data = {
            'states': [state],
            'actions': [action],
            'rewards': [],
            'traces': []
        }
        
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            next_state, reward, done = self.env.step(action)
            next_action = self.epsilon_greedy_policy(next_state, epsilon)
            
            # Standard SARSA update
            td_target = reward + self.gamma * self.Q[next_state][next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            episode_data['states'].append(next_state)
            episode_data['actions'].append(next_action)
            episode_data['rewards'].append(reward)
            episode_data['traces'].append(0)  # No traces
            
            state = next_state
            action = next_action
            steps += 1
        
        return episode_data


# ========================
# VISUALIZATION FUNCTIONS
# ========================

def plot_environment(env, ax, title=None):
    """Visualize environment"""
    if isinstance(env, SimpleGridWorld):
        grid = np.ones((env.size, env.size, 3)) * 0.95
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) == env.start:
                    grid[i, j] = [0.2, 0.8, 0.2]
                elif (i, j) == env.goal:
                    grid[i, j] = [1.0, 0.84, 0.0]
                elif (i, j) in env.obstacles:
                    grid[i, j] = [0.3, 0.3, 0.3]
        ax.imshow(grid, interpolation='nearest')
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
    elif isinstance(env, WindyGridWorld):
        grid = np.ones((env.height, env.width, 3)) * 0.95
        # Show wind strength with color
        for j in range(env.width):
            wind = env.wind[j]
            for i in range(env.height):
                if wind > 0:
                    grid[i, j] = [0.7, 0.9, 1.0 - wind*0.2]
        grid[env.start[0], env.start[1]] = [0.2, 0.8, 0.2]
        grid[env.goal[0], env.goal[1]] = [1.0, 0.84, 0.0]
        ax.imshow(grid, interpolation='nearest')
        ax.set_xticks(range(0, env.width, 2))
        ax.set_yticks(range(env.height))
        
    elif isinstance(env, CliffWalkingWorld):
        grid = np.ones((env.height, env.width, 3)) * 0.95
        for pos in env.cliff:
            grid[pos[0], pos[1]] = [0.8, 0.2, 0.2]
        grid[env.start[0], env.start[1]] = [0.2, 0.8, 0.2]
        grid[env.goal[0], env.goal[1]] = [1.0, 0.84, 0.0]
        ax.imshow(grid, interpolation='nearest')
        ax.set_xticks(range(0, env.width, 2))
        ax.set_yticks(range(env.height))
        
    elif isinstance(env, MazeWorld):
        grid = np.ones((env.size, env.size, 3)) * 0.95
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) == env.start:
                    grid[i, j] = [0.2, 0.8, 0.2]
                elif (i, j) == env.goal:
                    grid[i, j] = [1.0, 0.84, 0.0]
                elif (i, j) in env.obstacles:
                    grid[i, j] = [0.3, 0.3, 0.3]
                elif (i, j) in env.rewards:
                    grid[i, j] = [0.5, 0.8, 1.0]
        ax.imshow(grid, interpolation='nearest')
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
    
    ax.set_title(title or env.name, fontsize=10, fontweight='bold')
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)


def plot_learning_comparison(agents, ax, title="Learning Curves"):
    """Compare learning curves of different lambda values"""
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0] if data else 0
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    for idx, agent in enumerate(agents):
        color = colors[idx % len(colors)]
        episodes = range(len(agent.episode_returns))
        ax.plot(episodes, smooth(agent.episode_returns), 
               linewidth=2.5, label=agent.name, color=color, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Total Return', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_eligibility_traces(agent, ax, title="Eligibility Trace Evolution"):
    """Visualize how eligibility traces evolve"""
    if not agent.trace_history:
        ax.text(0.5, 0.5, 'No trace data available', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    for idx, traces in enumerate(agent.trace_history):
        steps = range(len(traces))
        alpha = 0.3 + 0.7 * (idx / len(agent.trace_history))
        ax.plot(steps, traces, alpha=alpha, linewidth=1.5,
               label=f'Episode {idx*20}')
    
    ax.set_xlabel('Step within Episode', fontsize=10)
    ax.set_ylabel('Max Eligibility Trace', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_final_performance(agents, ax):
    """Bar chart of final performance"""
    names = [agent.name for agent in agents]
    final_returns = [np.mean(agent.episode_returns[-50:]) for agent in agents]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    bars = ax.bar(names, final_returns, color=colors[:len(agents)], alpha=0.7)
    
    ax.set_ylabel('Average Return (Last 50 Episodes)', fontsize=10)
    ax.set_title('Final Performance Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)


def plot_lambda_comparison(agents, ax):
    """Show effect of different lambda values"""
    lambda_values = []
    convergence_speeds = []
    final_performances = []
    
    for agent in agents:
        if hasattr(agent, 'lambda_'):
            lambda_values.append(agent.lambda_)
            # Convergence speed: episodes to reach 80% of final performance
            final_perf = np.mean(agent.episode_returns[-50:])
            threshold = 0.8 * final_perf
            
            converged = False
            for i, ret in enumerate(agent.episode_returns):
                if ret >= threshold:
                    convergence_speeds.append(i)
                    converged = True
                    break
            if not converged:
                convergence_speeds.append(len(agent.episode_returns))
                
            final_performances.append(final_perf)
    
    if lambda_values:
        ax2 = ax.twinx()
        
        line1 = ax.plot(lambda_values, convergence_speeds, 'bo-', 
                       linewidth=2.5, markersize=8, label='Convergence Speed')
        line2 = ax2.plot(lambda_values, final_performances, 'rs-', 
                        linewidth=2.5, markersize=8, label='Final Performance')
        
        ax.set_xlabel('λ (Lambda)', fontsize=10)
        ax.set_ylabel('Episodes to Converge', fontsize=10, color='b')
        ax2.set_ylabel('Final Return', fontsize=10, color='r')
        ax.set_title('Effect of λ on Learning', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=9)


# ========================
# MAIN EXECUTION
# ========================

def run_td_lambda_experiments():
    """Run TD(λ) experiments on multiple environments"""
    
    print("="*90)
    print("TD(λ) WITH ELIGIBILITY TRACES - MULTI-ENVIRONMENT ANALYSIS")
    print("="*90)
    
    # Define environments
    environments = [
        SimpleGridWorld(),
        WindyGridWorld(),
        CliffWalkingWorld(),
        MazeWorld()
    ]
    
    # Lambda values to test
    lambda_values = [0.0, 0.3, 0.6, 0.9]
    
    results = {}
    
    for env_idx, env in enumerate(environments):
        print(f"\n{'='*90}")
        print(f"[{env_idx+1}] Environment: {env.name}")
        print(f"{'='*90}")
        
        agents = []
        
        # Train agents with different lambda values
        for lambda_val in lambda_values:
            print(f"\n  Training TD(λ={lambda_val})...", end=' ')
            
            if lambda_val == 0.0:
                agent = SARSA0Agent(env, alpha=0.2, gamma=0.95)
            else:
                agent = TDLambdaAgent(env, alpha=0.2, gamma=0.95, lambda_=lambda_val)
            
            agent.train(episodes=200, epsilon=0.1)
            agents.append(agent)
            
            final_return = np.mean(agent.episode_returns[-50:])
            final_length = np.mean(agent.episode_lengths[-50:])
            print(f"Final Return: {final_return:.2f}, Avg Length: {final_length:.2f}")
        
        results[env.name] = agents
    
    # Create comprehensive visualization
    print(f"\n{'='*90}")
    print("Generating visualizations...")
    print(f"{'='*90}")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Row 1: Environments
    for idx, env in enumerate(environments):
        ax = fig.add_subplot(gs[0, idx])
        plot_environment(env, ax)
    
    # Rows 2-5: Results for each environment
    for idx, (env_name, agents) in enumerate(results.items()):
        row = idx + 1
        
        # Learning curves
        ax1 = fig.add_subplot(gs[row, 0:2])
        plot_learning_comparison(agents, ax1, f"{env_name}: Learning Curves")
        
        # Final performance
        ax2 = fig.add_subplot(gs[row, 2])
        plot_final_performance(agents, ax2)
        
        # Lambda effect / Eligibility traces
        ax3 = fig.add_subplot(gs[row, 3])
        if len([a for a in agents if hasattr(a, 'lambda_')]) > 1:
            plot_lambda_comparison(agents, ax3)
        else:
            # Show traces for highest lambda
            high_lambda_agent = [a for a in agents if hasattr(a, 'lambda_') and a.lambda_ > 0.5]
            if high_lambda_agent:
                plot_eligibility_traces(high_lambda_agent[0], ax3)
    
    plt.suptitle('TD(λ) Algorithm: Multi-Environment Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('td_lambda_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'td_lambda_analysis.png'")
    plt.show()
    
    # Print detailed analysis
    print(f"\n{'='*90}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*90}\n")
    
    for env_name, agents in results.items():
        print(f"{env_name}:")
        print(f"  {'Agent':<15} {'Final Return':<15} {'Final Length':<15}")
        print(f"  {'-'*45}")
        for agent in agents:
            final_return = np.mean(agent.episode_returns[-50:])
            final_length = np.mean(agent.episode_lengths[-50:])
            print(f"  {agent.name:<15} {final_return:<15.2f} {final_length:<15.2f}")
        print()
    
    # Print algorithm explanation
    print(f"\n{'='*90}")
    print("TD(λ) ALGORITHM EXPLANATION")
    print(f"{'='*90}")
    print("""
TD(λ) Update Rule:
    For all states s and actions a:
        Q(s,a) ← Q(s,a) + α·δ·E(s,a)
        E(s,a) ← γ·λ·E(s,a)
    
    Where:
        δ = R + γ·Q(s',a') - Q(s,a)  (TD error)
        E(s,a) = Eligibility trace
    
Eligibility Traces:
    • Track which state-action pairs were recently visited
    • E(s,a) ← E(s,a) + 1 when (s,a) is visited
    • E(s,a) decays by γ·λ at each step
    • Updates propagate to ALL states with non-zero traces

Lambda (λ) Parameter:
    • λ = 0: TD(0) / SARSA(0) - Only immediate successor
    • λ = 1: Monte Carlo - Use complete episode return
    • 0 < λ < 1: Balance between TD and MC
    
    Effects of λ:
    ✓ Higher λ → Faster credit assignment (distant causes)
    ✓ Higher λ → More variance, potentially slower convergence
    ✓ Lower λ → More bias, bootstrapping from estimates
    ✓ Lower λ → Lower variance, potentially faster convergence

Key Advantages:
    1. CREDIT ASSIGNMENT: Rewards propagate backward efficiently
    2. FLEXIBILITY: λ controls bias-variance tradeoff
    3. SAMPLE EFFICIENCY: Better than TD(0) in many domains
    4. ONLINE LEARNING: Updates during episode, not just at end

When to use TD(λ):
    • Long episodes where delayed rewards are common
    • When quick credit assignment is important
    • Environments with clear cause-effect relationships
    • When you want to tune bias-variance tradeoff
    """)
    
    print(f"{'='*90}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*90}")


if __name__ == "__main__":
    run_td_lambda_experiments()