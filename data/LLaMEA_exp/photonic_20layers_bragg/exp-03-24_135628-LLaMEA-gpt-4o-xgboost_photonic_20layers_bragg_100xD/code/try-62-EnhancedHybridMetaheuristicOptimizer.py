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
            # Oppositional Learning for better exploration
            opposition_population = lb + ub - population
            opposition_fitness = np.array([func(ind) for ind in opposition_population])
            evaluations += len(opposition_fitness)
            combined_population = np.concatenate((population, opposition_population))
            combined_fitness = np.concatenate((fitness, opposition_fitness))
            best_opposition_index = np.argmin(combined_fitness)
            if combined_fitness[best_opposition_index] < best_fitness:
                best_fitness = combined_fitness[best_opposition_index]
                best_individual = combined_population[best_opposition_index]

            # Update population with best half
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices[:population_size]]
            fitness = combined_fitness[sorted_indices[:population_size]]

            # Dynamic mutation and crossover
            F = np.random.uniform(0.5, 1.0)
            CR = np.random.uniform(0.1, 0.9)
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
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

            # Adaptive Simulated Annealing
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(population_size):
                candidate = population[i] + np.random.normal(0, 0.1 * T, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = func(candidate)
                evaluations += 1
                if candidate_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - candidate_fitness) / T):
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness = candidate_fitness
                        best_individual = candidate

            if evaluations >= self.budget:
                break

        return best_individual