import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Adaptive Differential Evolution parameters
        F = 0.5  # Initial differential weight
        CR = 0.7  # Initial crossover probability

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adapt F and CR
                F = 0.5 + 0.3 * np.random.rand()
                CR = 0.6 + 0.4 * np.random.rand()

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Stochastic Hill Climbing
            if evaluations < self.budget:
                step_size = 0.1  # Hill climbing step size
                for i in range(population_size):
                    if evaluations >= self.budget:
                        break
                    current = population[i]
                    neighbor = np.clip(current + step_size * np.random.normal(size=self.dim), func.bounds.lb, func.bounds.ub)
                    neighbor_fitness = func(neighbor)
                    evaluations += 1

                    if neighbor_fitness < fitness[i]:
                        population[i] = neighbor
                        fitness[i] = neighbor_fitness

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]