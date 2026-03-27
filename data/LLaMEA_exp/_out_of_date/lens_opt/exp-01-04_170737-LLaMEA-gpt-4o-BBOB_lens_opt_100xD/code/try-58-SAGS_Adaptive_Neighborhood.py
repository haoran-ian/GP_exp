import numpy as np

class SAGS_Adaptive_Neighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.9   # Momentum factor for velocity update
        self.mutation_rate = 0.1  # Initial mutation rate
        self.population_size = 10
        self.best_position = None
        self.best_value = float('inf')
        self.inertia_weight = 0.9  # Initial inertia weight

    def __call__(self, func):
        # Initialize swarm positions, velocities, and local bests
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        local_best_positions = positions.copy()
        local_best_values = np.array([func(pos) for pos in positions])
        evaluations = self.population_size

        # Identify initial global best
        best_idx = np.argmin(local_best_values)
        self.best_value = local_best_values[best_idx]
        self.best_position = positions[best_idx].copy()

        # Dynamic learning rate, mutation, and inertia schedule
        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        inertia_schedule = lambda evals: self.inertia_weight * (1 - evals / self.budget)

        previous_best_value = self.best_value

        while evaluations < self.budget:
            # Update velocities with adaptive inertia and local best influence
            for i in range(self.population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)  # Random gradient approximation
                adaptive_alpha = alpha_schedule(evaluations)
                inertia = inertia_schedule(evaluations)
                velocities[i] = (
                    inertia * velocities[i]
                    + adaptive_alpha * gradient
                    + 0.2 * (self.best_position - positions[i])
                    + 0.1 * (local_best_positions[i] - positions[i])
                )

            # Update positions
            positions += velocities
            # Ensure positions are within bounds
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)

            # Evaluate new positions and update local bests
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1

                # Update local and global bests
                if value < local_best_values[i]:
                    local_best_values[i] = value
                    local_best_positions[i] = positions[i].copy()
                    if value < self.best_value:
                        self.best_value = value
                        self.best_position = positions[i].copy()

            # Elite selection and adaptive mutation
            elite_indices = local_best_values.argsort()[:self.population_size // 2]
            elites = positions[elite_indices]
            values_elites = local_best_values[elite_indices]
            for i in range(self.population_size // 2, self.population_size):
                if evaluations >= self.budget:
                    break
                parents = np.random.choice(elite_indices, 2, replace=False)
                offspring = 0.5 * (positions[parents[0]] + positions[parents[1]])
                mutation_strength = mutation_schedule(evaluations) * ((previous_best_value + 1e-9) / (self.best_value + 1e-9))
                offspring += np.random.normal(0, mutation_strength, self.dim)
                offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                value_offspring = func(offspring)
                evaluations += 1
                if value_offspring < local_best_values[i]:
                    positions[i] = offspring
                    local_best_values[i] = value_offspring
                    if value_offspring < self.best_value:
                        self.best_value = value_offspring
                        self.best_position = offspring.copy()

            # Refill the population with elites and new offspring
            positions[:self.population_size // 2] = elites
            local_best_values[:self.population_size // 2] = values_elites
            
            # Update previous best value for next iteration
            previous_best_value = self.best_value

        return self.best_position, self.best_value