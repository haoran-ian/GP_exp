import numpy as np

class SAGS_Synergistic_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.9   # Momentum factor for velocity update
        self.mutation_rate = 0.1  # Initial mutation rate
        self.population_size = 10
        self.best_position = None
        self.best_value = float('inf')
        self.neighborhood_size = 3  # Size of neighborhood for local learning

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
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)

        previous_best_value = self.best_value

        while evaluations < self.budget:
            # Update velocities with neighborhood-based learning
            for i in range(self.population_size):
                neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                neighborhood_best = positions[neighbors_indices[np.argmin(values[neighbors_indices])]]
                gradient = neighborhood_best - positions[i]
                adaptive_alpha = alpha_schedule(evaluations)
                diversity_factor = np.std(positions) / (func.bounds.ub - func.bounds.lb) * 10
                velocities[i] = (self.beta * velocities[i] 
                                 - adaptive_alpha * gradient * diversity_factor 
                                 + 0.2 * (self.best_position - positions[i]))

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

            # Elite selection
            elite_indices = values.argsort()[:self.population_size // 2]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]

            # Recombination and mutation
            for i in range(self.population_size // 2, self.population_size):
                if evaluations >= self.budget:
                    break
                parents = np.random.choice(elite_indices, 2, replace=False)
                offspring = 0.5 * positions[parents[0]] + 0.5 * positions[parents[1]]
                mutation_strength = mutation_schedule(evaluations) * ((previous_best_value + 1e-9) / (self.best_value + 1e-9))
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

            # Update previous best value for next iteration
            previous_best_value = self.best_value

        return self.best_position, self.best_value