import numpy as np

class HierarchicalAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.base_F = 0.5
        self.base_CR = 0.7
        self.entropy_threshold = 0.4
        self.adaptation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size

        while budget_spent < self.budget:
            # Calculate entropy of the current population
            fitness_mean = np.mean(fitness)
            entropy = -np.sum(np.log(np.abs(fitness - fitness_mean) + 1e-5))
            
            # Adjust F and CR based on entropy
            if entropy < self.entropy_threshold:
                F = self.base_F + self.adaptation_rate
                CR = self.base_CR - self.adaptation_rate
            else:
                F = self.base_F - self.adaptation_rate
                CR = self.base_CR + self.adaptation_rate
            
            # Hierarchical DE process
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x1 - x2), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if budget_spent >= self.budget:
                    break
        
        best_index = np.argmin(fitness)
        return population[best_index]