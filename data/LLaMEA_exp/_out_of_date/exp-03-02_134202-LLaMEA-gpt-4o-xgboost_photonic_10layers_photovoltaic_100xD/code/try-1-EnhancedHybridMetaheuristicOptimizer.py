import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 3 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.neighborhood_size = max(5, int(self.population_size * 0.1))
        self.energy_threshold = 0.01  # New parameter for energy-based local search

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Selection of random distinct indices
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation and Crossover
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):  # Ensure at least one crossover point
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial solution
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            # Dynamic Neighborhood Search with Energy-based Hill Climbing
            for i in range(self.population_size):
                neighborhood_indices = np.random.choice(
                    self.population_size, self.neighborhood_size, replace=False)
                neighborhood = population[neighborhood_indices]
                neighborhood_fitness = fitness[neighborhood_indices]
                
                local_best_idx = np.argmin(neighborhood_fitness)
                if evaluations >= self.budget:
                    break
                
                # Random perturbation in the local best
                perturbation = np.random.normal(0, 0.1, self.dim)
                candidate = neighborhood[local_best_idx] + perturbation
                candidate = np.clip(candidate, bounds[0], bounds[1])
                candidate_fitness = func(candidate)
                evaluations += 1

                # Energy-based acceptance criterion
                energy_diff = candidate_fitness - fitness[i]
                if energy_diff < self.energy_threshold:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

        # Return the best-found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]