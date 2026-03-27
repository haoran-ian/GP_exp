import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = min(50, self.budget // 2)
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        evaluations = population_size
        F_base, CR_base = 0.8, 0.9
        while evaluations < self.budget:
            # Adaptive DE Parameters
            F = F_base * (1 - evaluations / self.budget)
            CR = CR_base * (1 - evaluations / self.budget)
            for i in range(population_size):
                # DE mutation and crossover
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Function evaluation
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial

            # Diversified Simulated Annealing-like selection
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(population_size):
                candidate_shift = np.random.normal(0, 0.1 * T, self.dim)
                new_candidate = population[i] + candidate_shift
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                acceptance_probability = np.exp((fitness[i] - new_fitness) / T)
                if new_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = new_candidate

            if evaluations >= self.budget:
                break

        return best_individual