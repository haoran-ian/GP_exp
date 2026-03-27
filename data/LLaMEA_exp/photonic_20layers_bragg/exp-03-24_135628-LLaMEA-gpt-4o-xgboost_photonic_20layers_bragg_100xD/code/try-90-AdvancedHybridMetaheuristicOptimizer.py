import numpy as np

class AdvancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory = []  # Adaptive memory for storing elite solutions

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
            # Dynamic population-sizing strategy with adaptive memory influence
            current_pop_size = max(5, int(population_size * (1 - evaluations / self.budget)))
            F = np.random.uniform(0.6, 1.0)  # Adaptive mutation factor F
            CR = np.random.uniform(0.2, 0.9)  # Adaptive crossover rate CR
            
            for i in range(current_pop_size):
                # DE mutation and crossover with adaptive memory usage
                indices = [idx for idx in range(current_pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                if self.memory:
                    d = self.memory[np.random.randint(len(self.memory))]
                else:
                    d = population[np.random.choice(indices, 1, replace=False)][0]
                
                mutant = np.clip(a + F * (b - c + d - a), lb, ub)
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
                
                # Update memory with elite solutions
                if len(self.memory) < 5:  # Keep memory size manageable
                    self.memory.append(trial)
                else:
                    worst_index = np.argmax([func(mem) for mem in self.memory])
                    if trial_fitness < func(self.memory[worst_index]):
                        self.memory[worst_index] = trial

            # Simulated Annealing-like exploration
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(current_pop_size):
                new_candidate = population[i] + np.random.normal(0, 0.05, self.dim)  # Noise level
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