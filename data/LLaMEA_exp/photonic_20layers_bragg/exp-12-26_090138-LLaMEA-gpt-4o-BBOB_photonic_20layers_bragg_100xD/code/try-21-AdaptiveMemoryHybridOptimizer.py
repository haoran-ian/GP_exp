import numpy as np

class AdaptiveMemoryHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 15 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions
        best_mem = []
        F_adapt = 0.6
        CR_adapt = 0.8

        while eval_count < self.budget:
            sorted_indices = np.argsort(fitness)
            top_25_percent = sorted_indices[:population_size // 4]
            
            for i in range(population_size):
                # Self-adaptive parameter tuning
                F = F_adapt + np.random.uniform(-0.1, 0.1)
                CR = CR_adapt + np.random.uniform(-0.1, 0.1)

                # Mutation and crossover with memory-based enhancement
                indices = np.random.choice(top_25_percent, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate new candidate
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if len(best_mem) < 5 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:5]
                
                # Adaptive memory update
                if eval_count < self.budget and np.random.rand() < 0.1:
                    memory_candidate = np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                    f_memory_candidate = func(memory_candidate)
                    eval_count += 1
                    if f_memory_candidate < np.max(best_mem):
                        best_mem.append(f_memory_candidate)
                        best_mem = sorted(best_mem)[:5]

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]