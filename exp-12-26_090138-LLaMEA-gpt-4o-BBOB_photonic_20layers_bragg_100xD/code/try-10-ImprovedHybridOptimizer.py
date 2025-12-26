import numpy as np

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 10 * self.dim
        F_base = 0.5
        CR_base = 0.7
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        while eval_count < self.budget:
            for i in range(population_size):
                # Improved dynamic parameter adaptation
                F = F_base + np.random.randn() * 0.1
                CR = CR_base + np.random.rand() * 0.3  # Adjusted crossover range

                # Mutation and crossover with diversity preservation
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
                
                # Selection with fitness improvement check
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                # Adaptive local search with dynamic intensity
                if eval_count < self.budget and np.random.rand() < 0.3:  # Increased local search probability
                    perturbation_scale = 0.05 if f_trial < np.mean(fitness) else 0.2  # Adaptive perturbation based on fitness trend
                    perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    new_trial = trial + perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]