import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 10 * self.dim
        F = 0.8  # Initial differential weight
        CR = 0.9  # Initial crossover probability
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size
        
        # Track best fitness for dynamic parameter adjustment
        best_fitness = np.min(fitness)
        
        while eval_count < self.budget:
            for i in range(population_size):
                # Dynamic adjustment of F and CR
                F = 0.7 + 0.3 * np.random.rand()  # Randomize F for diversity
                CR = 0.8 + 0.2 * np.random.rand()  # Randomize CR for diversity
                
                # Mutation and crossover
                indices = np.random.choice(population_size, 3, replace=False)
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
                    # Update best fitness
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                
                # Adaptive Local Search
                if eval_count < self.budget and np.random.rand() < 0.2:  # 20% chance for local search
                    new_trial = trial + np.random.normal(0, 0.1, self.dim)
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if f_new_trial < best_fitness:
                            best_fitness = f_new_trial

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]