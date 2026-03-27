import numpy as np

class RefinedHybridMetaheuristicOptimizer:
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
        stagnation_counter = 0
        restart_threshold = 100  # Number of evaluations to trigger a restart if no improvement

        while evaluations < self.budget:
            # Dynamic population-sizing strategy
            current_pop_size = max(5, int(population_size * (1 - evaluations / self.budget)))
            F = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor F
            CR = np.random.uniform(0.1, 0.9)  # Adaptive crossover rate CR
            
            for i in range(current_pop_size):
                # DE mutation and crossover
                indices = [idx for idx in range(current_pop_size) if idx != i]
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
                    stagnation_counter = 0  # Reset stagnation counter

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                else:
                    stagnation_counter += 1

            # Enhanced Simulated Annealing-like selection
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(current_pop_size):
                new_candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness) / T):
                    population[i] = new_candidate
                    fitness[i] = new_fitness
                    stagnation_counter = 0  # Reset stagnation counter

                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_individual = new_candidate
                else:
                    stagnation_counter += 1

            # Adaptive Restart Mechanism
            if stagnation_counter >= restart_threshold:
                # Restart by reinitializing part of the population
                num_to_restart = max(1, current_pop_size // 5)
                restart_indices = np.random.choice(current_pop_size, num_to_restart, replace=False)
                for idx in restart_indices:
                    population[idx] = np.random.uniform(lb, ub, self.dim)
                    fitness[idx] = func(population[idx])
                    evaluations += 1
                stagnation_counter = 0

            if evaluations >= self.budget:
                break

        return best_individual