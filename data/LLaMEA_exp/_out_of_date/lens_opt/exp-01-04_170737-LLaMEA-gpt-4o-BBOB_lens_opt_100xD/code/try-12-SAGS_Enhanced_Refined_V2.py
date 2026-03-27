import numpy as np

class SAGS_Enhanced_Refined_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.85  # Momentum factor for velocity update
        self.initial_mutation_rate = 0.2  # Increased initial mutation rate
        self.min_mutation_rate = 0.05  # Minimum mutation rate
        self.population_size = 10
        self.best_position = None
        self.best_value = float('inf')

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

        # Dynamic learning rate and mutation schedule
        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: max(self.min_mutation_rate, self.initial_mutation_rate * (1 - evals / self.budget))

        while evaluations < self.budget:
            # Update velocities based on the adaptive gradient and momentum
            diversity_factor = np.std(positions, axis=0) / (func.bounds.ub - func.bounds.lb)
            adaptive_mutation = self.initial_mutation_rate * np.mean(diversity_factor)
            
            for i in range(self.population_size):
                gradient = np.random.normal(scale=adaptive_mutation, size=self.dim)  # Adaptive gradient approximation
                adaptive_alpha = alpha_schedule(evaluations)
                velocities[i] = self.beta * velocities[i] - adaptive_alpha * gradient * np.random.uniform(0.5, 1.5)
            
            # Update positions
            positions = positions + velocities
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
            
            # Elite selection: Retain top performers in the population
            elite_indices = values.argsort()[:self.population_size // 2]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]
            elite_performance = np.mean(values_elites)

            for i in range(self.population_size // 2, self.population_size):
                if evaluations >= self.budget:
                    break
                # Dynamic recombination and mutation for exploration
                parents = np.random.choice(elite_indices, 2, replace=False)
                recombination_weight = (values[parents[0]] + values[parents[1]]) / (2 * elite_performance)
                offspring = positions[parents[0]] * recombination_weight + positions[parents[1]] * (1 - recombination_weight)
                
                mutation_strength = mutation_schedule(evaluations)
                offspring += np.random.normal(0, mutation_strength, self.dim)
                offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                value_offspring = func(offspring)
                evaluations += 1
                if value_offspring < values[i]:
                    positions[i] = offspring
                    values[i] = value_offspring
                    if value_offspring < self.best_value:
                        self.best_value = value_offspring
                        self.best_position = offspring.copy()

            # Refill the population with elite and new offspring
            positions[:self.population_size // 2] = elites
            values[:self.population_size // 2] = values_elites

        return self.best_position, self.best_value