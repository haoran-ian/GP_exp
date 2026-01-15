import numpy as np

class SAGS_Diversity_Enhanced:
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
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        values = np.array([func(pos) for pos in positions])
        evaluations = self.population_size

        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()

        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        
        previous_best_value = self.best_value

        while evaluations < self.budget:
            for i in range(self.population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)
                adaptive_alpha = alpha_schedule(evaluations)
                diversity_factor = np.std(positions) / (func.bounds.ub - func.bounds.lb) * 10
                velocities[i] = self.beta * velocities[i] - adaptive_alpha * gradient * diversity_factor + 0.2 * (self.best_position - positions[i])
            
            positions = positions + velocities
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1
                
                if value < values[i]:
                    values[i] = value
                    if value < self.best_value:
                        self.best_value = value
                        self.best_position = positions[i].copy()
            
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

            positions[:self.population_size // 2] = elites
            values[:self.population_size // 2] = values_elites
            
            previous_best_value = self.best_value

        return self.best_position, self.best_value