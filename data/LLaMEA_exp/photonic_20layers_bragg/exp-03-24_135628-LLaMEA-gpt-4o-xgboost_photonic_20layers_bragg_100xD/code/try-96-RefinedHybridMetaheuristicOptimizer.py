import numpy as np

class RefinedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Chaotic initialization using logistic map
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(50, self.budget // 2)
        chaotic_seq = np.random.rand(population_size, self.dim)
        population = lb + (ub - lb) * chaotic_seq

        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        evaluations = population_size
        while evaluations < self.budget:
            # Adaptive population-sizing strategy
            current_pop_size = max(5, int(population_size * (1 - evaluations / self.budget)))
            F = np.random.uniform(0.5, 1.0)
            CR = np.random.uniform(0.1, 0.8)

            for i in range(current_pop_size):
                indices = [idx for idx in range(current_pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                d = population[np.random.choice(indices, 1, replace=False)][0]
                mutant = np.clip(a + F * (b - c + d - a), lb, ub)
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

            # Simulated Annealing with Lévy flight exploration
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(current_pop_size):
                levy_step = np.random.normal(0, 1, self.dim) * np.random.standard_cauchy(self.dim)
                new_candidate = population[i] + 0.01 * levy_step
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness) / T):
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = new_candidate

            if evaluations >= self.budget:
                break

        return best_individual