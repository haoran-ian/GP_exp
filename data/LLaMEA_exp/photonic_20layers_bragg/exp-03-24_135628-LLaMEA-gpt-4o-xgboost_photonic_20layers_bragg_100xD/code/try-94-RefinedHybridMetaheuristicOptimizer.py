import numpy as np

class RefinedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(50, self.budget // 2)
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        evaluations = population_size
        niche_radius = np.std(population, axis=0) * 0.1

        while evaluations < self.budget:
            current_pop_size = max(5, int(population_size * (1 - evaluations / self.budget)))
            F = np.random.uniform(0.6, 1.0)
            CR = np.random.uniform(0.2, 0.9)

            for i in range(current_pop_size):
                indices = [idx for idx in range(current_pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial

            T = max(0.01, 1.0 - evaluations / self.budget)
            step_size = np.clip(niche_radius * np.exp(-10 * evaluations / self.budget), 0.01, 0.1)

            for i in range(current_pop_size):
                new_candidate = population[i] + np.random.normal(0, step_size, self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness) / T):
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = new_candidate

            overlap_indices = []
            for i in range(current_pop_size):
                for j in range(i + 1, current_pop_size):
                    if np.linalg.norm(population[i] - population[j]) < niche_radius:
                        overlap_indices.append(j)
            population = np.delete(population, overlap_indices, axis=0)
            fitness = np.delete(fitness, overlap_indices)
            if evaluations >= self.budget:
                break

        return best_individual