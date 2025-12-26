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
        F_min, F_max = 0.5, 0.9  # Differential weight range
        CR_min, CR_max = 0.1, 0.9  # Crossover probability range

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Dynamic parameter adaptation
                F = F_min + np.random.rand() * (F_max - F_min)
                CR = CR_min + np.random.rand() * (CR_max - CR_min)

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

            # Local Search with Random Search
            if evaluations < self.budget:
                for _ in range(2):  # Two local perturbations
                    rand_idx = np.random.randint(0, population_size)
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    local_candidate = np.clip(population[rand_idx] + perturbation, func.bounds.lb, func.bounds.ub)
                    local_fitness = func(local_candidate)
                    evaluations += 1

                    if local_fitness < fitness[rand_idx]:
                        fitness[rand_idx] = local_fitness
                        population[rand_idx] = local_candidate

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]