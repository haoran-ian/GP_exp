import numpy as np

class SAGS_Adaptive_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1
        self.beta = 0.9
        self.mutation_rate = 0.1
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
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        
        previous_best_value = self.best_value

        while evaluations < self.budget:
            # Adaptive neighborhood size based on diversity
            diversity = np.std(positions, axis=0)
            neighborhood_size = 1 + int(self.population_size * (1 - np.mean(diversity) / (func.bounds.ub - func.bounds.lb)))

            # Update velocities based on adaptive gradient, momentum, and swarm topology
            for i in range(self.population_size):
                neighbors_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best_idx = neighbors_indices[np.argmin(values[neighbors_indices])]
                gradient = np.random.normal(scale=0.1, size=self.dim)
                adaptive_alpha = alpha_schedule(evaluations)
                velocities[i] = (self.beta * velocities[i] 
                                 - adaptive_alpha * gradient 
                                 + 0.2 * (positions[local_best_idx] - positions[i]) 
                                 + 0.2 * (self.best_position - positions[i]))

            # Update positions
            positions = positions + velocities
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
            
            # Elite selection and adaptive mutation
            elite_indices = values.argsort()[:self.population_size // 2]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]
            for i in range(self.population_size // 2, self.population_size):
                if evaluations >= self.budget:
                    break
                parents = np.random.choice(elite_indices, 2, replace=False)
                offspring = positions[parents[0]] * 0.5 + positions[parents[1]] * 0.5
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