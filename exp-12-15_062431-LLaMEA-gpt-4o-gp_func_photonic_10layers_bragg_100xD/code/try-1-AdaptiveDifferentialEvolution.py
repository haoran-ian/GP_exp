import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.eval_count = 0
        
    def __call__(self, func):
        # Initialize population
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), bounds[0], bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            
            # Adaptive parameter tuning
            self.crossover_rate = 0.5 + 0.3 * np.sin(np.pi * self.eval_count / self.budget)
            self.mutation_factor = 0.5 + 0.3 * np.cos(np.pi * (self.eval_count/self.budget) * (fitness.mean() / fitness.min()))  # Modified line

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]