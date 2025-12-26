import numpy as np

class RefinedHybridOptimizer:
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
        F_min, F_max = 0.5, 0.9  # Adaptive differential weight range
        CR_min, CR_max = 0.1, 0.9  # Adaptive crossover probability range

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation with adaptive F
                F = F_min + np.random.rand() * (F_max - F_min)
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover with adaptive CR
                CR = CR_min + np.random.rand() * (CR_max - CR_min)
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

            # Stochastic Local Search
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                local_search_point = population[best_idx] + np.random.normal(0, 0.1, self.dim)
                local_search_point = np.clip(local_search_point, func.bounds.lb, func.bounds.ub)
                local_fitness = func(local_search_point)
                evaluations += 1

                if local_fitness < fitness[best_idx]:
                    fitness[best_idx] = local_fitness
                    population[best_idx] = local_search_point
        
        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]