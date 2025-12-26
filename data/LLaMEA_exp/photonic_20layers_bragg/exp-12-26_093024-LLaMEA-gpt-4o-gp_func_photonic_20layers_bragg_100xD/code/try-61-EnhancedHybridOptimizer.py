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

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]

        # Adaptive DE parameters
        F_min, F_max = 0.5, 1.0  # Differential weight range
        CR_min, CR_max = 0.1, 0.9  # Crossover probability range

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adapt F and CR based on progress
                F = F_min + (F_max - F_min) * (1 - evaluations / self.budget)
                CR = CR_max - (CR_max - CR_min) * (1 - evaluations / self.budget)

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

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_idx = i

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

        # Return the best solution found
        return population[best_idx]