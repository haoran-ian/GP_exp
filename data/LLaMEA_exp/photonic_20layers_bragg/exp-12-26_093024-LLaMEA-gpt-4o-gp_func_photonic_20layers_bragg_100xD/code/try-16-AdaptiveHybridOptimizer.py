import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Differential Evolution parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        adaptive_threshold = 0.1

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

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

            # Adaptive Local Search
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                best_candidate = population[best_idx]

                # Localized search around the best candidate
                localized_points = np.random.uniform(
                    best_candidate - adaptive_threshold * (func.bounds.ub - func.bounds.lb),
                    best_candidate + adaptive_threshold * (func.bounds.ub - func.bounds.lb),
                    (population_size, self.dim)
                )
                localized_points = np.clip(localized_points, func.bounds.lb, func.bounds.ub)

                for local_point in localized_points:
                    if evaluations >= self.budget:
                        break
                    local_fitness = func(local_point)
                    evaluations += 1

                    if local_fitness < fitness[np.argmax(fitness)]:
                        worst_idx = np.argmax(fitness)
                        fitness[worst_idx] = local_fitness
                        population[worst_idx] = local_point

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]