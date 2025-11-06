import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time

class MDP:
    """Markov Decision Process Environment"""
    
    def __init__(self, env_type='grid'):
        self.environments = {
            'grid': {
                'name': 'Grid World MDP',
                'size': 4,
                'terminals': [(0, 0), (3, 3)],
                'obstacles': [(1, 1)],
                'rewards': {
                    (0, 0): 0,      # Terminal state
                    (3, 3): 1,      # Terminal state (goal)
                    (1, 1): -1,     # Obstacle
                    'default': -0.04  # Living penalty
                },
                'transition_prob': 0.8  # Probability of intended action
            },
            'cliff': {
                'name': 'Cliff Walking MDP',
                'size': (4, 12),
                'terminals': [(3, 11)],
                'obstacles': [(3, i) for i in range(1, 11)],  # The cliff
                'rewards': {
                    (3, 11): 1,     # Goal
                    **{(3, i): -1 for i in range(1, 11)},  # Cliff penalties
                    'default': -0.01
                },
                'transition_prob': 1.0  # Deterministic for cliff
            },
            'stochastic': {
                'name': 'Stochastic Grid MDP',
                'size': 5,
                'terminals': [(0, 4), (4, 4)],
                'obstacles': [(2, 2)],
                'rewards': {
                    (0, 4): -1,     # Bad terminal
                    (4, 4): 1,      # Good terminal
                    (2, 2): -0.5,   # Obstacle
                    'default': -0.02
                },
                'transition_prob': 0.7  # More stochastic
            }
        }
        
        config = self.environments[env_type]
        self.name = config['name']
        
        # Handle both square and rectangular grids
        if isinstance(config['size'], tuple):
            self.rows, self.cols = config['size']
        else:
            self.rows = self.cols = config['size']
        
        self.terminals = config['terminals']
        self.obstacles = config['obstacles']
        self.rewards = config['rewards']
        self.transition_prob = config['transition_prob']
        
        # Actions: up, down, left, right
        self.actions = ['↑', '↓', '←', '→']
        self.action_effects = {
            '↑': (-1, 0),
            '↓': (1, 0),
            '←': (0, -1),
            '→': (0, 1)
        }
        
        # Get all valid states
        self.states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.obstacles:
                    self.states.append((i, j))
    
    def get_reward(self, state):
        """Get reward for a state"""
        if state in self.rewards:
            return self.rewards[state]
        return self.rewards['default']
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return state in self.terminals
    
    def get_next_state(self, state, action):
        """Get next state given current state and action"""
        if self.is_terminal(state):
            return state
        
        effect = self.action_effects[action]
        next_state = (state[0] + effect[0], state[1] + effect[1])
        
        # Check boundaries and obstacles
        if (0 <= next_state[0] < self.rows and 
            0 <= next_state[1] < self.cols and
            next_state not in self.obstacles):
            return next_state
        return state  # Stay in place if invalid
    
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability P(s'|s,a)"""
        if self.is_terminal(state):
            return 1.0 if next_state == state else 0.0
        
        intended_next = self.get_next_state(state, action)
        
        if self.transition_prob == 1.0:  # Deterministic
            return 1.0 if next_state == intended_next else 0.0
        
        # Stochastic: might slip to perpendicular directions
        if next_state == intended_next:
            return self.transition_prob
        
        # Calculate perpendicular actions
        perpendicular_actions = []
        if action in ['↑', '↓']:
            perpendicular_actions = ['←', '→']
        else:
            perpendicular_actions = ['↑', '↓']
        
        # Check if next_state is reachable via perpendicular slip
        for perp_action in perpendicular_actions:
            if self.get_next_state(state, perp_action) == next_state:
                return (1 - self.transition_prob) / 2
        
        return 0.0


class PolicyIteration:
    """Policy Iteration Algorithm for MDP"""
    
    def __init__(self, mdp, gamma=0.9, theta=1e-6):
        self.mdp = mdp
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        
        # Initialize random policy
        self.policy = {}
        for state in mdp.states:
            if not mdp.is_terminal(state):
                self.policy[state] = np.random.choice(mdp.actions)
            else:
                self.policy[state] = None
        
        # Initialize value function
        self.V = {state: 0.0 for state in mdp.states}
        
        # Track iterations
        self.eval_iterations = []
        self.improvement_count = 0
        self.value_history = []
        self.policy_history = []
    
    def policy_evaluation(self, max_iterations=1000):
        """Evaluate current policy"""
        iteration = 0
        
        for iteration in range(max_iterations):
            delta = 0
            new_V = self.V.copy()
            
            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue
                
                action = self.policy[state]
                v = 0
                
                # Calculate expected value
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(next_state)
                    v += prob * (reward + self.gamma * self.V[next_state])
                
                new_V[state] = v
                delta = max(delta, abs(v - self.V[state]))
            
            self.V = new_V
            
            if delta < self.theta:
                break
        
        self.eval_iterations.append(iteration + 1)
        return iteration + 1
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue
            
            old_action = self.policy[state]
            
            # Find best action
            action_values = {}
            for action in self.mdp.actions:
                q_value = 0
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(next_state)
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                action_values[action] = q_value
            
            # Select best action (greedy)
            best_action = max(action_values, key=action_values.get)
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def iterate(self, max_iterations=100):
        """Run policy iteration until convergence"""
        for i in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"Policy Iteration {i + 1}")
            print(f"{'='*60}")
            
            # Policy Evaluation
            print("Evaluating policy...")
            eval_iters = self.policy_evaluation()
            print(f"  Converged in {eval_iters} iterations")
            
            # Store history
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())
            
            # Policy Improvement
            print("Improving policy...")
            policy_stable = self.policy_improvement()
            self.improvement_count += 1
            
            if policy_stable:
                print(f"\n✓ Policy converged after {i + 1} iterations!")
                break
        
        return self.policy, self.V


class PolicyIterationVisualizer:
    """Visualization for Policy Iteration"""
    
    def __init__(self, mdp, policy_iter):
        self.mdp = mdp
        self.policy_iter = policy_iter
        
    def visualize_results(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Policy Iteration for {self.mdp.name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Final Policy
        ax1 = fig.add_subplot(gs[0, 0])
        self.draw_policy(ax1, self.policy_iter.policy, 'Final Optimal Policy')
        
        # 2. Value Function Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self.draw_value_function(ax2, self.policy_iter.V, 'State Value Function')
        
        # 3. Statistics
        ax3 = fig.add_subplot(gs[0, 2])
        self.draw_statistics(ax3)
        
        # 4. Policy Evolution (first few iterations)
        iterations_to_show = min(3, len(self.policy_iter.policy_history))
        for i in range(iterations_to_show):
            ax = fig.add_subplot(gs[1, i])
            self.draw_policy(ax, self.policy_iter.policy_history[i], 
                           f'Policy Iteration {i+1}')
        
        # 5. Value Evolution
        for i in range(iterations_to_show):
            ax = fig.add_subplot(gs[2, i])
            self.draw_value_function(ax, self.policy_iter.value_history[i],
                                   f'Values Iteration {i+1}')
        
        plt.tight_layout()
        plt.show()
    
    def draw_policy(self, ax, policy, title):
        """Draw policy arrows on grid"""
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, self.mdp.cols)
        ax.set_ylim(0, self.mdp.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw grid
        for i in range(self.mdp.rows + 1):
            ax.axhline(i, color='gray', linewidth=0.5)
        for j in range(self.mdp.cols + 1):
            ax.axvline(j, color='gray', linewidth=0.5)
        
        # Draw cells
        for i in range(self.mdp.rows):
            for j in range(self.mdp.cols):
                state = (i, j)
                
                # Terminals
                if state in self.mdp.terminals:
                    color = 'lightgreen' if self.mdp.get_reward(state) > 0 else 'lightcoral'
                    rect = patches.Rectangle((j, i), 1, 1, 
                                            facecolor=color, alpha=0.7,
                                            edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    label = 'G' if self.mdp.get_reward(state) > 0 else 'T'
                    ax.text(j + 0.5, i + 0.5, label, 
                           ha='center', va='center', 
                           fontsize=14, fontweight='bold')
                
                # Obstacles
                elif state in self.mdp.obstacles:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            facecolor='gray', alpha=0.6,
                                            edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(j + 0.5, i + 0.5, '⛔',
                           ha='center', va='center', fontsize=16)
                
                # Regular states with policy
                elif state in policy and policy[state]:
                    action = policy[state]
                    ax.text(j + 0.5, i + 0.5, action,
                           ha='center', va='center',
                           fontsize=20, fontweight='bold', color='blue')
        
        ax.set_xticks(range(self.mdp.cols + 1))
        ax.set_yticks(range(self.mdp.rows + 1))
        ax.grid(True)
    
    def draw_value_function(self, ax, V, title):
        """Draw value function as heatmap"""
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Create value matrix
        value_matrix = np.zeros((self.mdp.rows, self.mdp.cols))
        for i in range(self.mdp.rows):
            for j in range(self.mdp.cols):
                state = (i, j)
                if state in V:
                    value_matrix[i, j] = V[state]
                else:
                    value_matrix[i, j] = np.nan
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('custom', 
                                                 ['red', 'yellow', 'green'])
        
        # Plot heatmap
        im = ax.imshow(value_matrix, cmap=cmap, aspect='auto')
        
        # Add value text
        for i in range(self.mdp.rows):
            for j in range(self.mdp.cols):
                state = (i, j)
                if state in V:
                    text = ax.text(j, i, f'{V[state]:.2f}',
                                 ha='center', va='center',
                                 fontsize=9, fontweight='bold')
                    
                    # Add background for readability
                    if state in self.mdp.obstacles:
                        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                       facecolor='gray', alpha=0.3))
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xticks(range(self.mdp.cols))
        ax.set_yticks(range(self.mdp.rows))
        ax.grid(True, alpha=0.3)
    
    def draw_statistics(self, ax):
        """Draw statistics and parameters"""
        ax.axis('off')
        
        stats_text = f"""
POLICY ITERATION RESULTS
{'='*40}

Total Iterations: {self.policy_iter.improvement_count}

Evaluation Iterations per Step:
{', '.join(map(str, self.policy_iter.eval_iterations[:5]))}
{'...' if len(self.policy_iter.eval_iterations) > 5 else ''}

Average V per Iteration:
"""
        
        for i, V in enumerate(self.policy_iter.value_history[:5]):
            avg_v = np.mean([v for v in V.values()])
            stats_text += f"  Iter {i+1}: {avg_v:.3f}\n"
        
        stats_text += f"""
{'='*40}
PARAMETERS

Discount Factor (γ): {self.policy_iter.gamma}
Convergence Threshold (θ): {self.policy_iter.theta}
Transition Probability: {self.mdp.transition_prob}

{'='*40}
LEGEND

→ ↓ ← ↑  : Policy actions
G        : Goal (terminal)
T        : Terminal state
⛔       : Obstacle
"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    print("="*70)
    print("POLICY ITERATION FOR MARKOV DECISION PROCESSES")
    print("="*70)
    print("\nThis program implements Policy Evaluation and Improvement")
    print("for solving MDPs using Dynamic Programming.\n")
    
    print("Available Environments:")
    print("1. Grid World MDP (4x4, stochastic)")
    print("2. Cliff Walking MDP (4x12, deterministic)")
    print("3. Stochastic Grid MDP (5x5, high uncertainty)")
    
    choice = input("\nSelect environment (1-3) [default: 1]: ").strip()
    env_map = {'1': 'grid', '2': 'cliff', '3': 'stochastic', '': 'grid'}
    env_type = env_map.get(choice, 'grid')
    
    # Get parameters
    gamma_input = input("Discount factor γ (0-1) [default: 0.9]: ").strip()
    gamma = float(gamma_input) if gamma_input else 0.9
    
    print(f"\nInitializing {env_type} MDP...")
    mdp = MDP(env_type)
    
    print(f"MDP Size: {mdp.rows}x{mdp.cols}")
    print(f"States: {len(mdp.states)}")
    print(f"Actions: {mdp.actions}")
    print(f"Transition Probability: {mdp.transition_prob}")
    
    print("\nStarting Policy Iteration...")
    policy_iter = PolicyIteration(mdp, gamma=gamma)
    
    # Run policy iteration
    optimal_policy, optimal_values = policy_iter.iterate()
    
    print("\n" + "="*70)
    print("POLICY ITERATION COMPLETE!")
    print("="*70)
    
    # Display some results
    print("\nSample Optimal Values:")
    for i, (state, value) in enumerate(list(optimal_values.items())[:10]):
        print(f"  V({state}) = {value:.4f}")
    
    print("\nSample Optimal Policy:")
    policy_items = [(s, a) for s, a in optimal_policy.items() if a is not None]
    for i, (state, action) in enumerate(policy_items[:10]):
        print(f"  π({state}) = {action}")
    
    print("\nGenerating visualizations...")
    visualizer = PolicyIterationVisualizer(mdp, policy_iter)
    visualizer.visualize_results()


if __name__ == "__main__":
    main()