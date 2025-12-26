import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(pop_size, self.dim)
        for i in range(self.dim):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        # Optimization loop
        while eval_count < self.budget:
            # Adaptive control parameters based on diversity and convergence rate
            F = 0.5 + (0.5 * np.std(fitness) / np.mean(fitness))
            CR = 0.5 + (0.5 * np.std(fitness) / np.mean(fitness))
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with fitness-weighted strategy
                best_index = np.argmin(fitness)
                x_best = population[best_index]
                indices = np.random.choice(pop_size, 4, replace=False)
                x0, x1, x2, x3 = population[indices]
                
                # Use fitness to weight the mutation strategy
                fitness_weight = fitness[indices] / np.sum(fitness[indices])
                mutant = np.clip(
                    x0 + F * (fitness_weight[0] * (x_best - x0) + fitness_weight[1] * (x1 - x2) + fitness_weight[2] * (x2 - x3)),
                    bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]