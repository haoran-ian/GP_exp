import numpy as np

class EnhancedHybridOptimizerV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 15 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions
        best_mem = [np.inf] * 5

        while eval_count < self.budget:
            for i in range(population_size):
                # Enhanced dynamic parameter adaptation
                adapt_factor = (1 - eval_count / self.budget) ** 0.5
                F = F_base + np.random.uniform(-0.1, 0.3) * adapt_factor
                CR = CR_base + np.random.uniform(-0.2, 0.2) * adapt_factor

                # Mutation and crossover with enhanced memory-based local search
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                oppo_mutant = bounds[:, 0] + bounds[:, 1] - mutant
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                oppo_trial = np.where(cross_points, oppo_mutant, population[i])
                
                # Evaluate new candidates
                f_trial = func(trial)
                f_oppo_trial = func(oppo_trial)
                eval_count += 2
                
                # Select the better trial
                if f_oppo_trial < f_trial:
                    trial = oppo_trial
                    f_trial = f_oppo_trial

                # Selection with fitness improvement check
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < max(best_mem):
                        best_mem.append(f_trial)
                        best_mem.remove(max(best_mem))

                # Memory-based local search with adaptive perturbation
                if eval_count < self.budget and np.random.rand() < 0.5:
                    perturbation_scale = 0.05 * np.random.uniform(0.5, 1.5) * adapt_factor
                    perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    new_trial = population[i] + perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if f_new_trial < max(best_mem):
                            best_mem.append(f_new_trial)
                            best_mem.remove(max(best_mem))

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]