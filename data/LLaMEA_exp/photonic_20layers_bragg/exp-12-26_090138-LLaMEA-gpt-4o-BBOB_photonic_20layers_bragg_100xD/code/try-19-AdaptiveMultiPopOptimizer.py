import numpy as np

class AdaptiveMultiPopOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        num_populations = 3
        population_size = 10 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        max_eval_per_pop = self.budget // num_populations

        # Create multiple populations
        populations = [np.random.rand(population_size, self.dim) for _ in range(num_populations)]
        populations = [bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0]) for pop in populations]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        eval_count = population_size * num_populations

        # Initialize adaptive learning rate
        learning_rate = 0.1

        while eval_count < self.budget:
            for p in range(num_populations):
                for i in range(population_size):
                    # Adaptive F and CR based on performance
                    F = F_base + np.random.uniform(-0.1, 0.3) * (1 - fitness[p][i] / np.max(fitness[p]))
                    CR = CR_base + np.random.uniform(-0.2, 0.2) * (1 - fitness[p][i] / np.max(fitness[p]))

                    # Mutation and crossover
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = populations[p][indices]
                    mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, populations[p][i])
                    
                    # Evaluate new candidate
                    f_trial = func(trial)
                    eval_count += 1

                    # Selection with adaptive learning
                    if f_trial < fitness[p][i]:
                        populations[p][i] = trial
                        fitness[p][i] = f_trial
                        learning_rate = max(0.01, learning_rate * 0.9)
                    else:
                        learning_rate = min(0.2, learning_rate * 1.1)

                # Inter-population migration
                if p < num_populations - 1 and eval_count < self.budget:
                    top_individuals = np.argsort(fitness[p])[:population_size // 5]
                    for idx in top_individuals:
                        target_pop = populations[p+1]
                        target_fitness = fitness[p+1]
                        if eval_count < self.budget:
                            target_pop[np.argmax(target_fitness)] = populations[p][idx]
                            target_fitness[np.argmax(target_fitness)] = fitness[p][idx]
                            eval_count += 1

        # Return the best solution found across all populations
        best_idx, best_fit = None, float('inf')
        for p in range(num_populations):
            idx = np.argmin(fitness[p])
            if fitness[p][idx] < best_fit:
                best_fit = fitness[p][idx]
                best_idx = (p, idx)
        return populations[best_idx[0]][best_idx[1]]