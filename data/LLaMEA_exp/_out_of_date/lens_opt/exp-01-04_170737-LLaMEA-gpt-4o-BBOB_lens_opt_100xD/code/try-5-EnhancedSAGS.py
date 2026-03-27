import numpy as np

class EnhancedSAGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_alpha = 0.1  # Initial learning rate for gradient-based update
        self.beta = 0.9           # Momentum factor for velocity update
        self.population_size = 10
        self.best_position = None
        self.best_value = float('inf')
        self.alpha_decay = 0.99   # Decay rate for learning rate

    def __call__(self, func):
        # Initialize swarm positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        
        # Evaluate initial positions
        values = np.array([func(pos) for pos in positions])
        evaluations = self.population_size
        
        # Identify initial best position
        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()
        
        alpha = self.initial_alpha
        
        while evaluations < self.budget:
            # Update velocities based on the adaptive gradient, momentum, and dynamic learning rate
            for i in range(self.population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)  # Random gradient approximation
                velocities[i] = (
                    self.beta * velocities[i] 
                    - alpha * gradient * np.random.uniform(0.5, 1.5)  # Adaptive step size
                )
                
                # Scale velocities adaptively based on current best known improvement
                velocity_scaling = 1 + np.tanh((self.best_value - values[i]) / self.best_value)
                velocities[i] *= velocity_scaling
            
            # Update positions
            positions += velocities
            # Ensure positions are within bounds
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)
            
            # Evaluate new positions
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1
                
                # Update the personal best and global best
                if value < values[i]:
                    values[i] = value
                    if value < self.best_value:
                        self.best_value = value
                        self.best_position = positions[i].copy()
            
            # Decay learning rate
            alpha *= self.alpha_decay

        return self.best_position, self.best_value