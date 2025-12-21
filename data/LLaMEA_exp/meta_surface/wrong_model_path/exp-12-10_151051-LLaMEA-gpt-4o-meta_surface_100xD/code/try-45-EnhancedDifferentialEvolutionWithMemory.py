import numpy as np

class EnhancedDifferentialEvolutionWithMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        memory_F = [0.5]
        memory_CR = [0.9]
        
        # Initialize population
        population = np.random.rand(pop_size, self.dim)
        for i in range(self.dim):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters with memory
            F = np.clip(np.random.normal(memory_F[-1], 0.1), 0.1, 1.0)
            CR = np.clip(np.random.normal(memory_CR[-1], 0.1), 0.1, 1.0)
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with adaptive memory strategy
                best_index = np.argmin(fitness)
                x_best = population[best_index]

                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
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
                    memory_F.append(F)
                    memory_CR.append(CR)
                    if len(memory_F) > 10:
                        memory_F.pop(0)
                        memory_CR.pop(0)

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]