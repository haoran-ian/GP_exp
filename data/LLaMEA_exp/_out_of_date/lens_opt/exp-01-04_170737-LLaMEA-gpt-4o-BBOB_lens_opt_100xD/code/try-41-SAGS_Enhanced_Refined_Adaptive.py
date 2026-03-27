import numpy as np

class SAGS_Enhanced_Refined_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.9   # Momentum factor for velocity update
        self.mutation_rate = 0.1  # Initial mutation rate
        self.initial_population_size = 10
        self.best_position = None
        self.best_value = float('inf')

    def __call__(self, func):
        # Initialize dynamic population size
        population_size = self.initial_population_size
        
        # Initialize swarm positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        
        # Evaluate initial positions
        values = np.array([func(pos) for pos in positions])
        evaluations = population_size
        
        # Identify initial best position
        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()

        # Dynamic learning rate and mutation schedule
        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        
        previous_best_value = self.best_value

        while evaluations < self.budget:
            # Update velocities based on the adaptive gradient and momentum
            for i in range(population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)  # Random gradient approximation
                adaptive_alpha = alpha_schedule(evaluations)
                diversity_factor = np.std(positions) / (func.bounds.ub - func.bounds.lb) * 10
                velocities[i] = self.beta * velocities[i] - adaptive_alpha * gradient * diversity_factor + 0.2 * (self.best_position - positions[i])
            
            # Update positions
            positions = positions + velocities
            # Ensure positions are within bounds
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)
            
            # Evaluate new positions
            for i in range(population_size):
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
            elite_indices = values.argsort()[:population_size // 2]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]
            for i in range(population_size // 2, population_size):
                if evaluations >= self.budget:
                    break
                # Recombination and mutation for exploration
                parents = np.random.choice(elite_indices, 2, replace=False)
                offspring = positions[parents[0]] * 0.5 + positions[parents[1]] * 0.5
                # Adaptive mutation strength based on convergence rate
                mutation_strength = mutation_schedule(evaluations) * (previous_best_value / (self.best_value + 1e-9))
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
            positions[:population_size // 2] = elites
            values[:population_size // 2] = values_elites
            
            # Dynamic population resizing based on convergence
            if evaluations < self.budget / 2:
                improvement_ratio = (previous_best_value - self.best_value) / (previous_best_value + 1e-9)
                if improvement_ratio < 0.01:
                    population_size = min(2 * population_size, self.budget - evaluations)
                    new_positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size - len(positions), self.dim))
                    positions = np.vstack((positions, new_positions))
                    velocities = np.vstack((velocities, np.zeros((population_size - len(velocities), self.dim))))
                    values = np.append(values, [float('inf')] * (population_size - len(values)))

            # Update previous best value for next iteration
            previous_best_value = self.best_value

        return self.best_position, self.best_value