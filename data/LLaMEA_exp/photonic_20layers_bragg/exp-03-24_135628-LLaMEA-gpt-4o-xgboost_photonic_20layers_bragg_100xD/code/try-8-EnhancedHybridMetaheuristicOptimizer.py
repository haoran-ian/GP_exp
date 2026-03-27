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
        while evaluations < self.budget:
            # Differential Evolution parameters with adaptive CR
            F, CR = 0.8, 0.7 + 0.3 * evaluations / self.budget
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

            # Enhanced Simulated Annealing with adaptive cooling
            T = max(0.01, 1.0 - (evaluations / self.budget) ** 0.5)
            for i in range(population_size):
                new_candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness) / T):
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = new_candidate

            # Local Search phase for refined exploitation
            for i in range(population_size):
                perturbed_candidate = best_individual + np.random.normal(0, 0.01, self.dim)
                perturbed_candidate = np.clip(perturbed_candidate, lb, ub)
                perturbed_fitness = func(perturbed_candidate)
                evaluations += 1
                if perturbed_fitness < best_fitness:
                    best_fitness = perturbed_fitness
                    best_individual = perturbed_candidate

            if evaluations >= self.budget:
                break

        return best_individual