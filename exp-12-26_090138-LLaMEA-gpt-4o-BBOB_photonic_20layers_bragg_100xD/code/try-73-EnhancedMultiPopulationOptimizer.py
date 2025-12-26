import numpy as np

class EnhancedMultiPopulationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        num_populations = 3
        population_size = 10 * self.dim
        F_base = 0.7
        CR_base = 0.9
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Create multiple populations
        populations = [np.random.rand(population_size, self.dim) for _ in range(num_populations)]
        populations = [bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0]) for pop in populations]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        eval_count = sum(population_size for _ in populations)
        
        # Memory for best solutions
        best_mem = []

        while eval_count < self.budget:
            for p_idx in range(num_populations):
                population = populations[p_idx]
                fitness = fitnesses[p_idx]
                
                for i in range(population_size):
                    # Adaptive parameter selection
                    F = F_base + np.random.uniform(-0.1, 0.1)
                    CR = CR_base + np.random.uniform(-0.1, 0.1)

                    # Mutation and crossover with stochastic ranking
                    if np.random.rand() < 0.5:
                        # DE/rand/1 scheme
                        indices = np.random.choice(population_size, 3, replace=False)
                        a, b, c = population[indices]
                        mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                    else:
                        # Rank-based selection
                        ranked_indices = np.argsort(fitness)
                        a, b, c = population[ranked_indices[:3]]
                        mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                    
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    
                    trial = np.where(cross_points, mutant, population[i])
                    
                    # Evaluate the trial candidate
                    f_trial = func(trial)
                    eval_count += 1

                    # Selection
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        if len(best_mem) < 5 or f_trial < np.max(best_mem):
                            best_mem.append(f_trial)
                            best_mem = sorted(best_mem)[:5]

                # Cooperation between populations
                if p_idx > 0 and np.random.rand() < 0.2:
                    top_idx = np.argmin(fitness)
                    populations[p_idx - 1][np.random.randint(0, population_size)] = population[top_idx]

        # Return the best solution found
        all_fitness = np.concatenate(fitnesses)
        all_population = np.concatenate(populations)
        best_idx = np.argmin(all_fitness)
        return all_population[best_idx]