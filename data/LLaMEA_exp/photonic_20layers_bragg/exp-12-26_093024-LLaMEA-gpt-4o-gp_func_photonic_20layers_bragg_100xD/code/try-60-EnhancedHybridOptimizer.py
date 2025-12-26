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
        F = 0.5  # Initial Differential weight
        CR = 0.9  # Initial Crossover probability

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)  # Track best solution index
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adapt parameters based on generation
                gen_factor = evaluations / self.budget
                F = 0.4 + 0.4 * gen_factor
                CR = 0.9 - 0.4 * gen_factor

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

            # Elitism: Preserve best solution
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

            # Ensure best solution is retained
            population[best_idx] = population[np.argmin(fitness)]
            fitness[best_idx] = min(fitness)

            # Local Search on the best solution to refine
            if evaluations < self.budget:
                local_search_candidate = population[best_idx] + np.random.normal(0, 0.1, self.dim)
                local_search_candidate = np.clip(local_search_candidate, func.bounds.lb, func.bounds.ub)
                local_fitness = func(local_search_candidate)
                evaluations += 1

                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_search_candidate
                    fitness[best_idx] = local_fitness

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]