import numpy as np

class AdaptiveHybridDifferentialEvolution:
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
        memory = np.zeros((5, self.dim))  # Memory to store successful mutations
        memory_idx = 0
        
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.5, 0.9)
            CR = np.random.uniform(0.2, 0.9)
            
            # Dynamic population resizing
            new_pop_size = max(5, int(pop_size * (1 - eval_count / self.budget)))
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            
            for i in range(new_pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with memory-enhanced strategy
                best_index = np.argmin(fitness)
                x_best = population[best_index]
                
                if eval_count > pop_size:
                    memory_indices = np.random.choice(min(len(memory), eval_count-pop_size), 3, replace=False)
                    x_mem = memory[memory_indices]
                    x0, x1, x2 = x_mem
                else:
                    indices = np.random.choice(new_pop_size, 3, replace=False)
                    x0, x1, x2 = [population[idx] for idx in indices]
                
                if np.random.rand() < 0.8:
                    mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                else:
                    mutant = np.clip(x0 + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
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
                    memory[memory_idx % len(memory)] = mutant
                    memory_idx += 1

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]