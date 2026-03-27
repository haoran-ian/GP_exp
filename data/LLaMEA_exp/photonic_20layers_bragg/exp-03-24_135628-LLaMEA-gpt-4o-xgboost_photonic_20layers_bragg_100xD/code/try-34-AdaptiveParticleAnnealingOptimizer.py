import numpy as np

class AdaptiveParticleAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(50, self.budget // 2)
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        p_best = population.copy()
        p_best_fitness = fitness.copy()
        best_index = np.argmin(fitness)
        g_best = population[best_index]
        g_best_fitness = fitness[best_index]

        evaluations = population_size
        w_max, w_min = 0.9, 0.4  # Inertia weights for PSO
        while evaluations < self.budget:
            # Dynamic inertia weight adaptation based on temperature-like mechanism
            T = max(0.01, 1.0 - evaluations / self.budget)
            w = w_max - (w_max - w_min) * (evaluations / self.budget)
            c1 = c2 = 2.0  # Cognitive and social coefficients

            # Update velocities and positions
            for i in range(population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (p_best[i] - population[i]) +
                                c2 * r2 * (g_best - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)

            # Evaluate fitness and update personal and global bests
            for i in range(population_size):
                fitness[i] = func(population[i])
                evaluations += 1
                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = population[i]
                    p_best_fitness[i] = fitness[i]
                if fitness[i] < g_best_fitness:
                    g_best = population[i]
                    g_best_fitness = fitness[i]

            # Simulated annealing-like exploration
            for i in range(population_size):
                new_candidate = population[i] + np.random.normal(0, T, self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness) / T):
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    if new_fitness < g_best_fitness:
                        g_best = new_candidate
                        g_best_fitness = new_fitness

            if evaluations >= self.budget:
                break

        return g_best