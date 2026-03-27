import numpy as np

class SAGS_Dynamic_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1
        self.beta = 0.9
        self.mutation_rate = 0.1
        self.initial_population_size = 10
        self.best_position = None
        self.best_value = float('inf')

    def __call__(self, func):
        population_size = self.initial_population_size
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        values = np.array([func(pos) for pos in positions])
        evaluations = population_size

        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()

        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        population_schedule = lambda evals: int(self.initial_population_size + (self.budget - evals) / self.budget * 5)

        while evaluations < self.budget:
            for i in range(population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)
                adaptive_alpha = alpha_schedule(evaluations)
                velocities[i] = self.beta * velocities[i] - adaptive_alpha * gradient * np.random.uniform(0.5, 1.5)

            positions = positions + velocities
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)

            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1
                if value < values[i]:
                    values[i] = value
                    if value < self.best_value:
                        self.best_value = value
                        self.best_position = positions[i].copy()

            elite_indices = values.argsort()[:population_size // 2]
            elites = positions[elite_indices]
            values_elites = values[elite_indices]

            for i in range(population_size // 2, population_size):
                if evaluations >= self.budget:
                    break
                parents = np.random.choice(elite_indices, 2, replace=False)
                weights = np.random.dirichlet(np.ones(2))
                offspring = weights[0] * positions[parents[0]] + weights[1] * positions[parents[1]]
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

            positions[:population_size // 2] = elites
            values[:population_size // 2] = values_elites

            population_size = max(5, population_schedule(evaluations))
            if positions.shape[0] != population_size:
                positions = np.resize(positions, (population_size, self.dim))
                velocities = np.resize(velocities, (population_size, self.dim))
                values = np.resize(values, population_size)

        return self.best_position, self.best_value