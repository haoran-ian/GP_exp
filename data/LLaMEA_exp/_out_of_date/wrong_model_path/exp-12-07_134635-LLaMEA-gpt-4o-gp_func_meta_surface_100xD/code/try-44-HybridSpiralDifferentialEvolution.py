import numpy as np

class HybridSpiralDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover rate for differential evolution
        self.f = 0.8   # Differential weight

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_value = fitness[best_index]

        # Spiral parameters
        radius = (self.upper_bound - self.lower_bound) / 2
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        inertia_weight = 0.9

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.f * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                
                # Spiral exploration
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.random.normal(0, 1, self.dim)
                direction /= np.linalg.norm(direction)
                spiral_candidate = population[i] + inertia_weight * radius * np.cos(angle) * direction
                spiral_candidate = np.clip(spiral_candidate, self.lower_bound, self.upper_bound)
                
                # Evaluate both candidates
                trial_value = func(trial)
                spiral_value = func(spiral_candidate)
                evaluations += 2

                # Choose the best among current, trial, and spiral
                if trial_value < fitness[i] and trial_value < spiral_value:
                    population[i] = trial
                    fitness[i] = trial_value
                elif spiral_value < fitness[i]:
                    population[i] = spiral_candidate
                    fitness[i] = spiral_value

                # Update global best
                if fitness[i] < best_value:
                    best_position = population[i]
                    best_value = fitness[i]
                    radius *= 0.8  # Contract radius on improvement
                    inertia_weight = max(0.4, inertia_weight - 0.01)
                else:
                    radius *= 1.1  # Expand radius for exploration
                    inertia_weight = min(0.9, inertia_weight + 0.01)

                # Ensure radius remains within sensible bounds
                radius = max(radius, min_radius)
                radius = min(radius, max_radius)

                if evaluations >= self.budget:
                    break

        return best_position, best_value