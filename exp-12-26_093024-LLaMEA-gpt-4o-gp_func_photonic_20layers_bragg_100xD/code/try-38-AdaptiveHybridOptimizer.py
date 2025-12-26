import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        F_min, F_max = 0.5, 1.0  # Adaptive Differential weight bounds
        CR_min, CR_max = 0.1, 0.9  # Adaptive Crossover probability bounds

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive Differential Evolution Parameters
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

            # Simulated Annealing as Random Search Replacement
            if evaluations < self.budget:
                temperature = max(0.1, 1.0 - evaluations / self.budget)  # Decreases over time
                current_idx = np.random.randint(0, population_size)
                current_sol = population[current_idx]
                current_fitness = fitness[current_idx]

                neighbor = current_sol + np.random.normal(0, temperature, self.dim)
                neighbor = np.clip(neighbor, func.bounds.lb, func.bounds.ub)

                neighbor_fitness = func(neighbor)
                evaluations += 1

                if neighbor_fitness < current_fitness or \
                   np.random.rand() < np.exp((current_fitness - neighbor_fitness) / temperature):
                    population[current_idx] = neighbor
                    fitness[current_idx] = neighbor_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]