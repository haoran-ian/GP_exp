import numpy as np

class ChaosEnhancedOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Define search space boundaries
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        while self.evaluations < self.budget:
            # Calculate chaos factor
            chaos_factor = self._chaos_map(self.evaluations / self.budget)
            
            # Generate offspring using chaos-enhanced strategy
            offspring = []
            for i in range(population_size):
                if np.random.rand() < chaos_factor:
                    # Chaos-driven Exploration
                    trial = self._chaos_perturbation(population[i], lb, ub)
                else:
                    # Adaptive Local Search
                    trial = self._adaptive_local_search(population, fitness, i, lb, ub, func)

                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                offspring.append((trial, trial_fitness))

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _chaos_perturbation(self, individual, lb, ub):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        perturbation *= np.random.rand(self.dim)
        trial = np.clip(individual + perturbation, lb, ub)
        return trial

    def _adaptive_local_search(self, population, fitness, index, lb, ub, func):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        direction = best_neighbor - population[index]
        trial = np.clip(population[index] + direction, lb, ub)
        return trial

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        neighbors = population[neighbor_indices]
        return neighbors

    def _chaos_map(self, t):
        # Implementing a simple logistic map for chaos generation
        r = 3.99  # Chaotic regime
        return r * t * (1 - t)