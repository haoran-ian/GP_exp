import numpy as np

class SAGS_Dynamic_Adaptive:
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
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        
        # Evaluate initial positions
        values = np.array([func(pos) for pos in positions])
        evaluations = population_size
        
        # Identify initial best position
        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()

        # Dynamic schedules
        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        inertia_weight_schedule = lambda evals: 0.9 - 0.5 * (evals / self.budget)

        while evaluations < self.budget:
            # Update velocities based on adaptive inertia weight
            inertia_weight = inertia_weight_schedule(evaluations)
            for i in range(population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)  # Random gradient approximation
                adaptive_alpha = alpha_schedule(evaluations)
                velocities[i] = inertia_weight * velocities[i] - adaptive_alpha * gradient * np.random.uniform(0.5, 1.5)
            
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
            elite_count = max(2, population_size // 4)
            elite_indices = values.argsort()[:elite_count]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]
            
            # Recombination and mutation for exploration
            new_population_size = max(2, elite_count * 2)
            new_positions = np.zeros((new_population_size, self.dim))
            new_values = np.full(new_population_size, float('inf'))
            for i in range(new_population_size):
                if evaluations >= self.budget:
                    break
                parents = np.random.choice(elite_indices, 2, replace=False)
                offspring = positions[parents[0]] * 0.5 + positions[parents[1]] * 0.5
                mutation_strength = mutation_schedule(evaluations)
                offspring += np.random.normal(0, mutation_strength, self.dim)
                offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                
                value_offspring = func(offspring)
                evaluations += 1
                new_positions[i] = offspring
                new_values[i] = value_offspring
                if value_offspring < self.best_value:
                    self.best_value = value_offspring
                    self.best_position = offspring.copy()

            # Update positions and values for the next generation
            positions = np.vstack((elites, new_positions))
            values = np.hstack((values_elites, new_values))
            population_size = len(positions)

        return self.best_position, self.best_value