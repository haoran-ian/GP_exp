import numpy as np

class HybridOptimizer:
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
        F_initial = 0.8  # Initial differential weight
        CR_initial = 0.9  # Initial crossover probability
        F = F_initial
        CR = CR_initial

        # Adaptive control parameters
        adaptation_rate = 0.05

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)  # Track best solution index
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
                    # Adaptive adjustment
                    F = min(1.0, F + adaptation_rate)
                    CR = min(1.0, CR + adaptation_rate)
                else:
                    # Adaptive adjustment
                    F = max(0.5, F - adaptation_rate)
                    CR = max(0.3, CR - adaptation_rate)

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
        best_idx = np.argmin(fitness)
        return population[best_idx]